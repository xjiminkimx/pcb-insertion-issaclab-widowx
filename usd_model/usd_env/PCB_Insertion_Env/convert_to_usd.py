#!/usr/bin/env python3
"""
Convert PCB_Insertion_Env URDF + STL meshes to an IsaacSim-compatible USDA fixture.

Generates a kinematic static fixture (magazine + guide rails) with:
  - Correct world-space mesh positions computed via URDF forward kinematics (FK).
  - UsdPreviewSurface materials (light-blue plastic for magazine, brushed metal for rails).
  - PhysicsCollisionAPI + PhysicsMeshCollisionAPI on every mesh so the PCB rigid body
    can slide and contact the rails and magazine slot correctly in PhysX.

The chip / PCB link is intentionally excluded — it is spawned separately as a
RigidObjectCfg (CuboidCfg) in widowx_pcb_env_cfg.py.

Usage (standalone, no pxr required):
    python convert_to_usd.py
Output: ../pcb_insertion_env.usd  (relative to this script's directory)

The file overwrites any existing pcb_insertion_env.usd.
"""

from __future__ import annotations

import math
import os
import struct
from collections import defaultdict, deque
from pathlib import Path
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rpy_to_mat3(rpy: tuple) -> list:
    """RPY (roll, pitch, yaw) → 3×3 rotation matrix (row-major list of lists)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    Rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    return mat3_mul(mat3_mul(Rz, Ry), Rx)


def mat3_mul(A: list, B: list) -> list:
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat3_vec(R: list, v: tuple) -> tuple:
    return tuple(sum(R[i][k] * v[k] for k in range(3)) for i in range(3))


def make_transform(xyz: tuple, rpy: tuple) -> tuple:
    """Return (R3x3, t3) representing a rigid transform."""
    return rpy_to_mat3(rpy), xyz


def compose_transforms(T_parent, T_child) -> tuple:
    """Compose (R_p, t_p) ∘ (R_c, t_c) → (R_out, t_out)."""
    R_p, t_p = T_parent
    R_c, t_c = T_child
    R_out = mat3_mul(R_p, R_c)
    t_c_rot = mat3_vec(R_p, t_c)
    t_out = tuple(t_p[i] + t_c_rot[i] for i in range(3))
    return R_out, t_out


IDENTITY_T = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0.0, 0.0, 0.0))


def apply_transform(T: tuple, v: tuple) -> tuple:
    """Apply (R, t) to a 3-vector."""
    R, t = T
    rv = mat3_vec(R, v)
    return (rv[0] + t[0], rv[1] + t[1], rv[2] + t[2])


def mat3_to_quat_xyzw(R: list) -> tuple:
    """3×3 rotation matrix → quaternion (x, y, z, w)."""
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2][1] - R[1][2]) * s
        y = (R[0][2] - R[2][0]) * s
        z = (R[1][0] - R[0][1]) * s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + R[0][0] - R[1][1] - R[2][2]))
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + R[1][1] - R[0][0] - R[2][2]))
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + R[2][2] - R[0][0] - R[1][1]))
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    return (x, y, z, w)


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def parse_urdf(urdf_path: str) -> tuple[dict, list]:
    """Return (links_info, joints_list).

    links_info[name] = {mesh, visual_pos, visual_rpy, material_color}
    joints_list = [{name, type, parent, child, pos, rpy, axis}]
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: dict = {}
    for link in root.findall("link"):
        name = link.get("name")
        visual = link.find("visual")
        info: dict = {"mesh": None, "visual_pos": (0.0, 0.0, 0.0),
                      "visual_rpy": (0.0, 0.0, 0.0),
                      "material_color": (0.7, 0.7, 0.7)}
        if visual is not None:
            geo = visual.find("geometry/mesh")
            if geo is not None:
                info["mesh"] = geo.get("filename")
            origin = visual.find("origin")
            if origin is not None:
                info["visual_pos"] = tuple(map(float, origin.get("xyz", "0 0 0").split()))
                info["visual_rpy"] = tuple(map(float, origin.get("rpy", "0 0 0").split()))
            mat = visual.find("material/color")
            if mat is not None:
                rgba = list(map(float, mat.get("rgba", "0.7 0.7 0.7 1").split()))
                info["material_color"] = tuple(rgba[:3])
        links[name] = info

    joints: list = []
    for jt in root.findall("joint"):
        origin = jt.find("origin")
        pos = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
        if origin is not None:
            pos = tuple(map(float, origin.get("xyz", "0 0 0").split()))
            rpy = tuple(map(float, origin.get("rpy", "0 0 0").split()))
        axis_elem = jt.find("axis")
        ax = (0.0, 0.0, 1.0)
        if axis_elem is not None:
            ax = tuple(map(float, axis_elem.get("xyz", "0 0 1").split()))
        parent_link = jt.find("parent")
        child_link = jt.find("child")
        if parent_link is None or child_link is None:
            continue
        joints.append({
            "name": jt.get("name"),
            "type": jt.get("type", "fixed"),
            "parent": parent_link.get("link"),
            "child": child_link.get("link"),
            "pos": pos,
            "rpy": rpy,
            "axis": ax,
        })

    return links, joints


def compute_fk(joints: list, root_link: str = "root") -> dict:
    """BFS from root_link, composing transforms at q=0 for all joints.

    Returns {link_name: (R3x3, t3)} in root frame.
    Loop-closure / duplicate children are visited once (first encountered wins).
    """
    # Build adjacency (parent → [(child, joint_info)])
    children: dict = defaultdict(list)
    for j in joints:
        children[j["parent"]].append(j)

    transforms: dict = {root_link: IDENTITY_T}
    queue = deque([root_link])
    visited: set = {root_link}

    while queue:
        parent = queue.popleft()
        T_parent = transforms[parent]
        for j in children[parent]:
            child = j["child"]
            if child in visited:
                continue
            visited.add(child)
            # Joint transform at q=0: only the static origin offset
            T_joint = make_transform(j["pos"], j["rpy"])
            T_child = compose_transforms(T_parent, T_joint)
            transforms[child] = T_child
            queue.append(child)

    return transforms


# ---------------------------------------------------------------------------
# STL reading
# ---------------------------------------------------------------------------

def read_stl_binary(path: str) -> list[tuple]:
    """Read binary STL → list of triangles [(v0,v1,v2), ...]."""
    triangles = []
    try:
        with open(path, "rb") as f:
            header = f.read(80)
            # ASCII detection
            if b"solid" in header[:6]:
                return read_stl_ascii(path)
            num_tri = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_tri):
                f.read(12)  # skip normal
                v0 = struct.unpack("<fff", f.read(12))
                v1 = struct.unpack("<fff", f.read(12))
                v2 = struct.unpack("<fff", f.read(12))
                f.read(2)
                triangles.append((v0, v1, v2))
    except Exception as e:
        print(f"  Warning: binary STL read failed ({e}), trying ASCII…")
        return read_stl_ascii(path)
    return triangles


def read_stl_ascii(path: str) -> list[tuple]:
    """Read ASCII STL → list of triangles."""
    triangles = []
    verts: list = []
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("vertex"):
                    parts = line.split()
                    verts.append(tuple(map(float, parts[1:4])))
                    if len(verts) == 3:
                        triangles.append(tuple(verts))
                        verts = []
    except Exception as e:
        print(f"  Warning: ASCII STL read failed ({e})")
    return triangles


def get_mesh_path(meshes_dir: str, mesh_ref: str) -> str:
    if "package://" in mesh_ref:
        mesh_file = mesh_ref.split("meshes/")[-1]
        return os.path.join(meshes_dir, mesh_file)
    return mesh_ref


# ---------------------------------------------------------------------------
# Geometry: transform and simplify
# ---------------------------------------------------------------------------

def transform_triangles(triangles: list, T: tuple) -> list:
    """Apply transform T=(R,t) to every vertex of every triangle."""
    return [
        tuple(apply_transform(T, v) for v in tri)
        for tri in triangles
    ]


def subsample_triangles(triangles: list, max_tris: int = 8000) -> list:
    """Evenly subsample if mesh is too large (keeps USD files manageable)."""
    if len(triangles) <= max_tris:
        return triangles
    step = max(1, len(triangles) // max_tris)
    return triangles[::step]


def triangles_to_arrays(triangles: list) -> tuple[list, list]:
    """Deduplicate vertices → (points, faceVertexIndices)."""
    points: list = []
    idx_map: dict = {}
    indices: list = []
    for tri in triangles:
        for v in tri:
            key = (round(v[0], 7), round(v[1], 7), round(v[2], 7))
            if key not in idx_map:
                idx_map[key] = len(points)
                points.append(key)
            indices.append(idx_map[key])
    return points, indices


# ---------------------------------------------------------------------------
# USDA generation helpers
# ---------------------------------------------------------------------------

USDA_HEADER = """\
#usda 1.0
(
    defaultPrim = "PCB_Env"
    doc = "PCB Insertion Environment - magazine + guide rails (kinematic fixture)"
    metersPerUnit = 1
    timeCodesPerSecond = 24
    upAxis = "Z"
)

"""

MATERIAL_TEMPLATE = """\
        def Material "{name}"
        {{
            token outputs:surface.connect = <{root}/Looks/{name}/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({r:.6f}, {g:.6f}, {b:.6f})
                float inputs:roughness = {roughness:.3f}
                float inputs:metallic = {metallic:.3f}
                token outputs:surface
            }}
        }}
"""

MESH_TEMPLATE = """\
        def Mesh "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"]
        )
        {{
            uniform bool doubleSided = 1
            int[] faceVertexCounts = [{counts}]
            int[] faceVertexIndices = [{indices}]
            point3f[] points = [{points}]
            rel material:binding = <{root}/Looks/{mat}>
            uniform token physics:approximation = "{approx}"
            float physxCollision:contactOffset = 0.004
            float physxCollision:restOffset = 0.0012
        }}
"""


def format_points(pts: list) -> str:
    return ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in pts)


def format_ints(lst: list) -> str:
    return ", ".join(str(i) for i in lst)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

# Links to include in the static fixture (exclude chip/PCB and loop-closure dummies)
STATIC_LINKS = {
    "magazine",
    "part_1",
    "part_1_2",
    "part_1_3",
    "part_1_5",
    "part_1_6",
    "part_1_7",
    "part_1_8",
    "part_1_9",
    # part_1_1 / part_1_4 / part_1_10 / part_1_11 are tiny connectors — include for completeness
    "part_1_1",
    "part_1_4",
    "part_1_10",
    "part_1_11",
}

# Magazine body only; every other static link uses RailMaterial (identical guide-rail look).
MAGAZINE_LINKS = {"magazine"}

# Collision approximation per group
APPROX_MAP = {
    "magazine": "convexDecomposition",   # concave slot → needs decomposition
    "part_1": "convexDecomposition",
    "part_1_2": "convexHull",
    "part_1_3": "convexHull",
    "part_1_5": "convexHull",
    "part_1_6": "convexHull",
    "part_1_7": "convexHull",
    "part_1_8": "convexHull",
    "part_1_9": "convexHull",
    "part_1_1": "convexHull",
    "part_1_4": "convexHull",
    "part_1_10": "convexHull",
    "part_1_11": "convexHull",
}


def create_pcb_insertion_env_usd(urdf_path: str, meshes_dir: str, output_path: str):
    print(f"Parsing URDF: {urdf_path}")
    links, joints = parse_urdf(urdf_path)

    print("Computing forward kinematics (all joints at q=0)…")
    fk = compute_fk(joints, root_link="root")

    # -----------------------------------------------------------------------
    # Build per-link world-space mesh content
    # -----------------------------------------------------------------------
    root_path = "/PCB_Env"
    mesh_blocks: list[str] = []
    bbox_all: list = []

    for link_name in STATIC_LINKS:
        info = links.get(link_name)
        if info is None or info["mesh"] is None:
            continue

        stl_path = get_mesh_path(meshes_dir, info["mesh"])
        if not os.path.exists(stl_path):
            print(f"  [skip] mesh not found: {stl_path}")
            continue

        print(f"  Processing {link_name}: {os.path.basename(stl_path)}")
        triangles = read_stl_binary(stl_path)
        if not triangles:
            print(f"    [skip] no triangles read")
            continue

        # Full world transform = T_link_in_root ∘ T_visual_origin
        T_link = fk.get(link_name, IDENTITY_T)
        T_visual = make_transform(info["visual_pos"], info["visual_rpy"])
        T_world = compose_transforms(T_link, T_visual)

        # Transform vertices to root frame
        triangles_world = transform_triangles(triangles, T_world)
        triangles_world = subsample_triangles(triangles_world, max_tris=8000)

        points, indices = triangles_to_arrays(triangles_world)
        bbox_all.extend(points)

        pts_str = format_points(points)
        idx_str = format_ints(indices)
        n_tris = len(indices) // 3
        counts_str = format_ints([3] * n_tris)

        mat_name = "MagazineMaterial" if link_name in MAGAZINE_LINKS else "RailMaterial"
        approx = APPROX_MAP.get(link_name, "convexDecomposition")
        safe_name = link_name.replace("_", "_")  # already safe

        mesh_blocks.append(MESH_TEMPLATE.format(
            name=safe_name,
            counts=counts_str,
            indices=idx_str,
            points=pts_str,
            root=root_path,
            mat=mat_name,
            approx=approx,
        ))

    # -----------------------------------------------------------------------
    # Report assembly bounding box for env_cfg tuning
    # -----------------------------------------------------------------------
    if bbox_all:
        xs = [p[0] for p in bbox_all]
        ys = [p[1] for p in bbox_all]
        zs = [p[2] for p in bbox_all]
        print(f"\nAssembly bounding box in USD root frame:")
        print(f"  X: [{min(xs):.4f}, {max(xs):.4f}]  (extent {max(xs)-min(xs):.4f} m)")
        print(f"  Y: [{min(ys):.4f}, {max(ys):.4f}]  (extent {max(ys)-min(ys):.4f} m)")
        print(f"  Z: [{min(zs):.4f}, {max(zs):.4f}]  (extent {max(zs)-min(zs):.4f} m)")

        # Rail top Z (for PCB spawn height reference)
        mag_link = "magazine"
        mag_info = links.get(mag_link)
        if mag_info:
            T_mag = compose_transforms(fk.get(mag_link, IDENTITY_T),
                                       make_transform(mag_info["visual_pos"], mag_info["visual_rpy"]))
            mag_center = T_mag[1]
            print(f"\nMagazine visual center (in USD root frame): "
                  f"({mag_center[0]:.4f}, {mag_center[1]:.4f}, {mag_center[2]:.4f})")
            print(f"  → Set env_cfg _MAG_POS to this value (after applying world rotation/translation).")

    # -----------------------------------------------------------------------
    # Assemble USDA
    # -----------------------------------------------------------------------
    mag_mat = MATERIAL_TEMPLATE.format(
        name="MagazineMaterial", root=root_path,
        r=0.615686, g=0.811765, b=0.929412, roughness=0.55, metallic=0.05)
    # One shared rail look for every guide-rail mesh (matte hard plastic).
    rail_mat = MATERIAL_TEMPLATE.format(
        name="RailMaterial", root=root_path,
        r=0.16, g=0.17, b=0.19, roughness=0.78, metallic=0.0)

    meshes_joined = "\n".join(mesh_blocks)

    content = USDA_HEADER + f"""\
def Xform "PCB_Env" (
    kind = "component"
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
)
{{
    bool physics:kinematicEnabled = true

    def Scope "Looks"
    {{
{mag_mat}
{rail_mat}
    }}

    def Xform "Geometry"
    {{
{meshes_joined}
    }}
}}
"""

    with open(output_path, "w") as fout:
        fout.write(content)

    print(f"\nWrote USD: {output_path}")
    print(f"Total mesh prims: {len(mesh_blocks)}")


def main():
    script_dir = Path(__file__).parent
    urdf_path = script_dir / "urdf" / "assembly_1.urdf"
    meshes_dir = script_dir / "meshes"
    # Output alongside existing pcb_insertion_env.usd in usd_model/usd_env/
    output_usd = script_dir.parent / "pcb_insertion_env.usd"

    if not urdf_path.exists():
        print(f"ERROR: URDF not found: {urdf_path}")
        return

    create_pcb_insertion_env_usd(str(urdf_path), str(meshes_dir), str(output_usd))

    print("\n--- env_cfg notes ---")
    print("  The USD root frame corresponds to the URDF 'root' link origin.")
    print("  Adjust _MAG_POS / _MAG_ROT_WXYZ in widowx_pcb_env_cfg.py to")
    print("  place the assembly at the desired world location.")
    print("  The PCB (chip link) is NOT in this USD — spawn it via RigidObjectCfg.")


if __name__ == "__main__":
    main()
