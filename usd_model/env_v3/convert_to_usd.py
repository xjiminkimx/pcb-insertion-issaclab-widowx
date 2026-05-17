#!/usr/bin/env python3
"""
Convert env_v3 URDF (assembly_2) to an IsaacSim-compatible USDA kinematic fixture.

Three-material classification:
  SteelMaterial   — magazine body                   (metallic steel, mu_s 0.55)
  RailMaterial    — guide-rail bars (Part_1_2.stl)  (hard plastic,   mu_s 0.80)
  StandMaterial   — stand + conveyor structure       (brushed steel,  mu_s 0.50)

All mesh vertices are FK-baked into the URDF "root" frame at q=0.
Chip / PCB link is intentionally excluded — spawned separately as CuboidCfg.

Output: ../usd_env/pcb_insertion_env.usd  (overwrites existing fixture)

Usage (standalone):
    cd usd_model/env_v3 && python3 convert_to_usd.py
"""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rpy_to_mat3(rpy: tuple) -> list:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    Rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    return mat3_mul(mat3_mul(Rz, Ry), Rx)

def mat3_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def mat3_vec(R, v):
    return tuple(sum(R[i][k] * v[k] for k in range(3)) for i in range(3))

def make_transform(xyz, rpy):
    return rpy_to_mat3(rpy), xyz

def compose_transforms(T_parent, T_child):
    R_p, t_p = T_parent
    R_c, t_c = T_child
    R_out = mat3_mul(R_p, R_c)
    t_c_rot = mat3_vec(R_p, t_c)
    t_out = tuple(t_p[i] + t_c_rot[i] for i in range(3))
    return R_out, t_out

IDENTITY_T = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0.0, 0.0, 0.0))

def apply_transform(T, v):
    R, t = T
    rv = mat3_vec(R, v)
    return (rv[0] + t[0], rv[1] + t[1], rv[2] + t[2])

# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def parse_urdf(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        visual = link.find("visual")
        if visual is not None:
            origin = visual.find("origin")
            xyz = tuple(float(x) for x in (origin.get("xyz", "0 0 0")).split())
            rpy = tuple(float(x) for x in (origin.get("rpy", "0 0 0")).split())
            geom = visual.find("geometry")
            mesh_el = geom.find("mesh") if geom is not None else None
            mesh_file = mesh_el.get("filename") if mesh_el is not None else None
        else:
            xyz, rpy, mesh_file = (0, 0, 0), (0, 0, 0), None
        links[name] = {"visual_pos": xyz, "visual_rpy": rpy, "mesh": mesh_file}

    joints = {}
    for joint in root.findall("joint"):
        jname = joint.get("name")
        jtype = joint.get("type")
        origin = joint.find("origin")
        if origin is not None:
            xyz = tuple(float(x) for x in (origin.get("xyz", "0 0 0")).split())
            rpy = tuple(float(x) for x in (origin.get("rpy", "0 0 0")).split())
        else:
            xyz, rpy = (0, 0, 0), (0, 0, 0)
        parent = joint.find("parent").get("link")
        child  = joint.find("child").get("link")
        joints[jname] = {
            "type": jtype, "xyz": xyz, "rpy": rpy,
            "parent": parent, "child": child,
        }
    return links, joints

# ---------------------------------------------------------------------------
# Forward kinematics at q=0 (prismatic/revolute/continuous contributions = 0)
# ---------------------------------------------------------------------------

def compute_fk(joints, root_link="root"):
    parent_map = {}  # child_link -> (parent_link, joint_transform)
    for jdata in joints.values():
        T_joint = make_transform(jdata["xyz"], jdata["rpy"])
        parent_map[jdata["child"]] = (jdata["parent"], T_joint)

    link_T = {root_link: IDENTITY_T}

    def resolve(link):
        if link in link_T:
            return link_T[link]
        if link not in parent_map:
            link_T[link] = IDENTITY_T
            return IDENTITY_T
        parent, T_joint = parent_map[link]
        T_parent = resolve(parent)
        T_link = compose_transforms(T_parent, T_joint)
        link_T[link] = T_link
        return T_link

    for link in list(parent_map.keys()):
        resolve(link)
    return link_T

# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def get_mesh_path(meshes_dir: str, mesh_ref: str) -> str:
    if "package://" in mesh_ref:
        mesh_file = mesh_ref.split("meshes/")[-1]
        return os.path.join(meshes_dir, mesh_file)
    return mesh_ref

def read_stl_binary(path: str) -> list:
    triangles = []
    with open(path, "rb") as f:
        f.read(80)  # header
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            f.read(12)  # normal
            verts = [struct.unpack("<fff", f.read(12)) for _ in range(3)]
            f.read(2)   # attr
            triangles.append(tuple(verts))
    return triangles

def transform_triangles(triangles, T):
    return [tuple(apply_transform(T, v) for v in tri) for tri in triangles]

def subsample_triangles(triangles, max_tris=10000):
    if len(triangles) <= max_tris:
        return triangles
    step = max(1, len(triangles) // max_tris)
    return triangles[::step]

def triangles_to_arrays(triangles):
    points, idx_map, indices = [], {}, []
    for tri in triangles:
        for v in tri:
            key = (round(v[0], 7), round(v[1], 7), round(v[2], 7))
            if key not in idx_map:
                idx_map[key] = len(points)
                points.append(key)
            indices.append(idx_map[key])
    return points, indices

def format_points(pts):
    return ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in pts)

def format_ints(lst):
    return ", ".join(str(i) for i in lst)

# ---------------------------------------------------------------------------
# USDA templates
# ---------------------------------------------------------------------------

USDA_HEADER = """\
#usda 1.0
(
    defaultPrim = "PCB_Env"
    doc = "PCB Insertion Environment v3 - magazine + guide rails + stand (kinematic fixture)"
    metersPerUnit = 1
    timeCodesPerSecond = 24
    upAxis = "Z"
)

"""

# Physics material: visual + PhysicsMaterialAPI + PhysxMaterialAPI on the material prim,
# MaterialBindingAPI on each mesh prim with both visual and physics binding.
MATERIAL_TEMPLATE = """\
        def Material "{name}" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            token outputs:surface.connect = <{root}/Looks/{name}/Shader.outputs:surface>
            float physics:staticFriction    = {static_friction:.3f}
            float physics:dynamicFriction   = {dynamic_friction:.3f}
            float physics:restitution       = {restitution:.3f}
            uniform token physxMaterial:frictionCombineMode     = "multiply"
            uniform token physxMaterial:restitutionCombineMode  = "multiply"

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({r:.6f}, {g:.6f}, {b:.6f})
                float inputs:roughness      = {roughness:.3f}
                float inputs:metallic       = {metallic:.3f}
                token outputs:surface
            }}
        }}
"""

MESH_TEMPLATE = """\
        def Mesh "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI",
                                  "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            uniform bool doubleSided = 1
            int[] faceVertexCounts   = [{counts}]
            int[] faceVertexIndices  = [{indices}]
            point3f[] points         = [{points}]
            rel material:binding         = <{root}/Looks/{mat}>
            rel material:binding:physics = <{root}/Looks/{mat}>
            uniform token physics:approximation = "{approx}"
            float physxCollision:contactOffset  = 0.004
            float physxCollision:restOffset     = 0.0012
        }}
"""

# ---------------------------------------------------------------------------
# Link classification
# ---------------------------------------------------------------------------

# Exclude: chip (spawned separately as CuboidCfg), loop-closure / planar dummies
STATIC_LINKS = {
    "magazine",
    "part_1",
    "part_1_1",
    "part_1_2",
    "part_1_3",
    "part_1_4",
    "part_1_5",
    "part_1_6",
    "part_1_7",
    "part_1_8",
    "part_1_9",
    "part_1_10",
    "part_1_11",
    "part_1_12",
    "part_1_13",
    "part_1_14",
}

# Guide-rail bars (Part_1_2.stl) → hard plastic
GUIDE_RAIL_LINKS = {"part_1_2", "part_1_10"}

# Magazine body → steel
MAGAZINE_LINKS = {"magazine"}

# All others → structural steel stand / conveyor
# (STATIC_LINKS - GUIDE_RAIL_LINKS - MAGAZINE_LINKS)

# Collision approximation per link
APPROX_MAP = {
    "magazine":   "convexDecomposition",   # concave slot pocket
    "part_1_7":   "convexDecomposition",   # large conveyor platform
    "part_1_12":  "convexDecomposition",   # main stand body
    "part_1_1":   "convexHull",
    "part_1_6":   "convexHull",
}
DEFAULT_APPROX = "convexHull"

def mat_for(link_name: str) -> str:
    if link_name in MAGAZINE_LINKS:
        return "SteelMaterial"
    if link_name in GUIDE_RAIL_LINKS:
        return "RailMaterial"
    return "StandMaterial"

# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def create_usd(urdf_path: str, meshes_dir: str, output_path: str):
    print(f"Parsing URDF: {urdf_path}")
    links, joints = parse_urdf(urdf_path)

    print("Computing FK at q=0 …")
    fk = compute_fk(joints, root_link="root")

    root_path = "/PCB_Env"
    mesh_blocks: list[str] = []
    bbox_all: list = []

    for link_name in sorted(STATIC_LINKS):
        info = links.get(link_name)
        if info is None or info["mesh"] is None:
            print(f"  [skip] no visual: {link_name}")
            continue

        stl_path = get_mesh_path(meshes_dir, info["mesh"])
        if not os.path.exists(stl_path):
            print(f"  [skip] STL not found: {stl_path}")
            continue

        print(f"  {link_name}: {os.path.basename(stl_path)}  [{mat_for(link_name)}]")
        triangles = read_stl_binary(stl_path)
        if not triangles:
            print(f"    [skip] no triangles")
            continue

        T_link   = fk.get(link_name, IDENTITY_T)
        T_visual = make_transform(info["visual_pos"], info["visual_rpy"])
        T_world  = compose_transforms(T_link, T_visual)

        tris = transform_triangles(triangles, T_world)
        tris = subsample_triangles(tris, max_tris=10000)
        points, indices = triangles_to_arrays(tris)
        bbox_all.extend(points)

        n_tris = len(indices) // 3
        mat    = mat_for(link_name)
        approx = APPROX_MAP.get(link_name, DEFAULT_APPROX)

        mesh_blocks.append(MESH_TEMPLATE.format(
            name=link_name,
            counts=format_ints([3] * n_tris),
            indices=format_ints(indices),
            points=format_points(points),
            root=root_path,
            mat=mat,
            approx=approx,
        ))

    # -----------------------------------------------------------------------
    # Bounding box report for env_cfg tuning
    # -----------------------------------------------------------------------
    if bbox_all:
        xs = [p[0] for p in bbox_all]
        ys = [p[1] for p in bbox_all]
        zs = [p[2] for p in bbox_all]
        print(f"\nAssembly bounding box in USD root frame:")
        print(f"  X: [{min(xs):.4f}, {max(xs):.4f}]  extent {max(xs)-min(xs):.4f} m")
        print(f"  Y: [{min(ys):.4f}, {max(ys):.4f}]  extent {max(ys)-min(ys):.4f} m")
        print(f"  Z: [{min(zs):.4f}, {max(zs):.4f}]  extent {max(zs)-min(zs):.4f} m")

    # Chip / PCB FK reference for env_cfg tuning
    chip_info = links.get("chip")
    if chip_info:
        T_chip   = fk.get("chip", IDENTITY_T)
        T_vis    = make_transform(chip_info["visual_pos"], chip_info["visual_rpy"])
        T_total  = compose_transforms(T_chip, T_vis)
        chip_ctr = T_total[1]
        print(f"\nchip visual centre in USD root frame: "
              f"({chip_ctr[0]:.4f}, {chip_ctr[1]:.4f}, {chip_ctr[2]:.4f})")
        print(f"  With _MAG_ROT_WXYZ=-90°Z and _MAG_POS=(tx,ty,tz) world pos ≈")
        print(f"    x = {chip_ctr[1]:.4f} + tx  (root-Y maps to world-X)")
        print(f"    y = {-chip_ctr[0]:.4f} + ty  (root-X maps to world -Y, inverted)")
        print(f"    z = {chip_ctr[2]:.4f} + tz")

    # -----------------------------------------------------------------------
    # Build materials
    # -----------------------------------------------------------------------
    # magazine: brushed steel (silver, highly metallic)
    steel_mat = MATERIAL_TEMPLATE.format(
        name="SteelMaterial", root=root_path,
        r=0.58, g=0.60, b=0.63,
        roughness=0.22, metallic=0.92,
        static_friction=0.55, dynamic_friction=0.42, restitution=0.02,
    )
    # guide rail bars: hard dark plastic
    rail_mat = MATERIAL_TEMPLATE.format(
        name="RailMaterial", root=root_path,
        r=0.16, g=0.17, b=0.19,
        roughness=0.78, metallic=0.0,
        static_friction=0.80, dynamic_friction=0.62, restitution=0.02,
    )
    # stand / conveyor structure: matt steel (slightly darker than magazine)
    stand_mat = MATERIAL_TEMPLATE.format(
        name="StandMaterial", root=root_path,
        r=0.45, g=0.47, b=0.50,
        roughness=0.35, metallic=0.85,
        static_friction=0.50, dynamic_friction=0.38, restitution=0.02,
    )

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
{steel_mat}
{rail_mat}
{stand_mat}
    }}

    def Xform "Geometry"
    {{
{meshes_joined}
    }}
}}
"""

    with open(output_path, "w") as fout:
        fout.write(content)

    print(f"\nWrote {output_path}")
    print(f"Mesh prims: {len(mesh_blocks)}")
    print("\n--- env_cfg reminders ---")
    print("  steel  (magazine):          SteelMaterial  mu_s=0.55")
    print("  plastic (guide rails):      RailMaterial   mu_s=0.80")
    print("  steel  (stand/conveyor):    StandMaterial  mu_s=0.50")
    print("  Tune _MAG_POS / _RAIL_SURFACE_Z in Isaac Sim after loading.")


def main():
    script_dir = Path(__file__).parent
    urdf_path  = script_dir / "urdf" / "assembly_2.urdf"
    meshes_dir = script_dir / "meshes"
    output_usd = script_dir.parent / "usd_env" / "pcb_insertion_env.usd"

    if not urdf_path.exists():
        print(f"ERROR: URDF not found: {urdf_path}")
        return
    if not meshes_dir.exists():
        print(f"ERROR: meshes dir not found: {meshes_dir}")
        return

    create_usd(str(urdf_path), str(meshes_dir), str(output_usd))


if __name__ == "__main__":
    main()
