#!/usr/bin/env python3
"""
Convert env_v6 URDF (assembly_6) to an IsaacSim-compatible USDA kinematic fixture.

Material + physics settings:
  SteelMaterial   — magazine body (incl. slot)  mu_s 0.80  contactOffset 0.0001  restOffset 0.0001
  RailMaterial    — guide-rail bars (Part_1_7.stl)  mu_s 0.10
  BeltMaterial    — side conveyor belts (Part_1_2.stl)  mu_s 0.90
  StandMaterial   — stand + frame            mu_s 0.50

Collision approximations:
  Magazine / guide rails / side belts / side rail-guides — Triangle Mesh (``none``).
  Stand / frame                                         — convexDecomposition (decorative contact).
Short axle rods (Part_1_3.stl) and chip/PCB link excluded — chip spawned separately.

Recommended Physics Scene settings in env cfg:
  sim.dt = 1/480 or 1/240   (480 Hz or 240 Hz)

Output: ./pcb_insertion_env.usd  (same directory as this script, env_v6/)

Usage:
    cd usd_model/env_v6 && python3 convert_to_usd.py
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
# Forward kinematics at q=0
# ---------------------------------------------------------------------------

def compute_fk(joints, root_link="root"):
    parent_map = {}
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
        f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
        for _ in range(count):
            f.read(12)
            verts = [struct.unpack("<fff", f.read(12)) for _ in range(3)]
            f.read(2)
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

def mesh_basename(mesh_ref: str | None) -> str:
    if not mesh_ref:
        return ""
    return os.path.basename(mesh_ref.split("meshes/")[-1])

# ---------------------------------------------------------------------------
# USDA templates
# ---------------------------------------------------------------------------

USDA_HEADER = """\
#usda 1.0
(
    defaultPrim = "PCB_Env"
    doc = "PCB Insertion Environment v6 - magazine (fine slot) + guide rails + side belts + stand (kinematic fixture)"
    metersPerUnit = 1
    timeCodesPerSecond = 24
    upAxis = "Z"
)

"""

MATERIAL_TEMPLATE = """\
        def Material "{name}" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            token outputs:surface.connect = <{root}/Looks/{name}/Shader.outputs:surface>
            float physics:staticFriction    = {static_friction:.4f}
            float physics:dynamicFriction   = {dynamic_friction:.4f}
            float physics:restitution       = {restitution:.4f}
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

# Stand / frame — convex decomposition is fine (not a precision support surface).
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
            uniform token physics:approximation = "convexDecomposition"
            float physxCollision:contactOffset  = 0.0001
            float physxCollision:restOffset     = 0.0001
        }}
"""

# Magazine / rails / belts — Triangle Mesh (``none``) so flat support faces and the
# magazine slot match the visual STL (kinematic fixture; safe with dynamic PCB cuboid).
MESH_TEMPLATE_TRIANGLE = """\
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
            uniform token physics:approximation = "none"
            float physxCollision:contactOffset  = 0.0001
            float physxCollision:restOffset     = 0.0001
        }}
"""

# ---------------------------------------------------------------------------
# Link classification
# ---------------------------------------------------------------------------

CHIP_LINKS = {"root", "chip"}
MAGAZINE_LINKS = {"magazine"}
SHORT_AXLE_MESH = "Part_1_3.stl"    # short axle rods — decorative, excluded
# Tall vertical guides beside the conveyor (included in fixture for lane walls).
SIDE_RAIL_GUIDE_MESHES = frozenset({"Part_1_4.stl", "Part_1_6.stl"})
BELT_MESH = "Part_1_2.stl"          # side conveyor belts
RAIL_MESH = "Part_1_7.stl"          # guide-rail bars (horizontal PCB support)


def export_links(links: dict) -> list[str]:
    out = []
    for name, info in links.items():
        if not info.get("mesh") or name in CHIP_LINKS:
            continue
        base = mesh_basename(info["mesh"])
        if base == SHORT_AXLE_MESH:
            continue
        out.append(name)
    return sorted(out)


def belt_links(links: dict) -> set[str]:
    return {
        name for name, info in links.items()
        if mesh_basename(info.get("mesh")) == BELT_MESH
    }


def uses_triangle_mesh(link_name: str, mesh_ref: str | None) -> bool:
    """True for magazine, guide rails, side belts, and side rail-guides."""
    if link_name in MAGAZINE_LINKS:
        return True
    base = mesh_basename(mesh_ref)
    return base in (BELT_MESH, RAIL_MESH) or base in SIDE_RAIL_GUIDE_MESHES


def mat_for(link_name: str, mesh_ref: str | None = None) -> str:
    if link_name in MAGAZINE_LINKS:
        return "SteelMaterial"
    base = mesh_basename(mesh_ref)
    if base == BELT_MESH:
        return "BeltMaterial"
    if base == RAIL_MESH or base in SIDE_RAIL_GUIDE_MESHES:
        return "RailMaterial"
    return "StandMaterial"


def link_mesh_bbox(link_name: str, links: dict, fk: dict, meshes_dir: str):
    info = links.get(link_name)
    if not info or not info.get("mesh"):
        return None
    stl_path = get_mesh_path(meshes_dir, info["mesh"])
    if not os.path.exists(stl_path):
        return None
    triangles = read_stl_binary(stl_path)
    T_link = fk.get(link_name, IDENTITY_T)
    T_vis = make_transform(info["visual_pos"], info["visual_rpy"])
    T_world = compose_transforms(T_link, T_vis)
    tris = transform_triangles(triangles, T_world)
    pts = [p for tri in tris for p in tri]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def create_usd(urdf_path: str, meshes_dir: str, output_path: str):
    print(f"Parsing URDF: {urdf_path}")
    links, joints = parse_urdf(urdf_path)

    print("Computing FK at q=0 …")
    fk = compute_fk(joints, root_link="part_1_1")
    static_links = export_links(links)
    side_belts = belt_links(links)
    print(f"  exporting {len(static_links)} mesh links")
    print(f"  side belt links: {', '.join(sorted(side_belts))}")

    root_path = "/PCB_Env"
    mesh_blocks: list[str] = []
    bbox_all: list = []
    mag_points: list = []

    for link_name in static_links:
        info = links.get(link_name)
        if info is None or info["mesh"] is None:
            continue

        stl_path = get_mesh_path(meshes_dir, info["mesh"])
        if not os.path.exists(stl_path):
            print(f"  [skip] STL not found: {stl_path}")
            continue

        is_mag = link_name in MAGAZINE_LINKS
        use_tri = uses_triangle_mesh(link_name, info["mesh"])
        mat = mat_for(link_name, info["mesh"])
        approx_tag = "triangleMesh" if use_tri else "convexDecomp"
        print(f"  {link_name}: {os.path.basename(stl_path)}  [{mat}]  [{approx_tag}]")

        triangles = read_stl_binary(stl_path)
        if not triangles:
            print(f"    [skip] no triangles")
            continue

        T_link   = fk.get(link_name, IDENTITY_T)
        T_visual = make_transform(info["visual_pos"], info["visual_rpy"])
        T_world  = compose_transforms(T_link, T_visual)

        tris = transform_triangles(triangles, T_world)
        # Precision surfaces keep full STL; stand/frame may subsample for USD size.
        if use_tri:
            print(f"    preserving all {len(tris)} triangles (triangle-mesh collider)")
        else:
            tris = subsample_triangles(tris, max_tris=10000)

        points, indices = triangles_to_arrays(tris)
        bbox_all.extend(points)
        if is_mag:
            mag_points.extend(points)

        n_tris = len(indices) // 3
        tmpl = MESH_TEMPLATE_TRIANGLE if use_tri else MESH_TEMPLATE

        mesh_blocks.append(tmpl.format(
            name=link_name,
            counts=format_ints([3] * n_tris),
            indices=format_ints(indices),
            points=format_points(points),
            root=root_path,
            mat=mat,
        ))

    slot_mouth_x = None
    if bbox_all:
        xs = [p[0] for p in bbox_all]
        ys = [p[1] for p in bbox_all]
        zs = [p[2] for p in bbox_all]
        slot_mouth_x = min(xs)
        print(f"\nAssembly bounding box in USD root frame:")
        print(f"  X: [{min(xs):.4f}, {max(xs):.4f}]  extent {max(xs)-min(xs):.4f} m")
        print(f"  Y: [{min(ys):.4f}, {max(ys):.4f}]  extent {max(ys)-min(ys):.4f} m")
        print(f"  Z: [{min(zs):.4f}, {max(zs):.4f}]  extent {max(zs)-min(zs):.4f} m")

    if mag_points:
        mxs = [p[0] for p in mag_points]
        mys = [p[1] for p in mag_points]
        mzs = [p[2] for p in mag_points]
        mcx = 0.5 * (min(mxs) + max(mxs))
        mcy = 0.5 * (min(mys) + max(mys))
        mcz = 0.5 * (min(mzs) + max(mzs))
        print(f"\nMagazine bounding box in USD root frame:")
        print(f"  X: [{min(mxs):.4f}, {max(mxs):.4f}]  extent {max(mxs)-min(mxs):.4f} m")
        print(f"  Y: [{min(mys):.4f}, {max(mys):.4f}]  extent {max(mys)-min(mys):.4f} m")
        print(f"  Z: [{min(mzs):.4f}, {max(mzs):.4f}]  extent {max(mzs)-min(mzs):.4f} m")
        print(f"  centre (root): ({mcx:.4f}, {mcy:.4f}, {mcz:.4f})")

    for link_name in sorted(side_belts):
        bb = link_mesh_bbox(link_name, links, fk, meshes_dir)
        if bb:
            print(f"\nside belt top Z in USD root frame ({link_name}): {bb[5]:.4f}")

    chip_link = "chip" if links.get("chip", {}).get("mesh") else "root"
    chip_info = links.get(chip_link)
    if chip_info and chip_info.get("mesh"):
        bb = link_mesh_bbox(chip_link, links, fk, meshes_dir)
        T_chip = fk.get(chip_link, IDENTITY_T)
        T_vis = make_transform(chip_info["visual_pos"], chip_info["visual_rpy"])
        chip_ctr = compose_transforms(T_chip, T_vis)[1]
        print(f"\nchip visual centre in USD root frame ({chip_link}): "
              f"({chip_ctr[0]:.4f}, {chip_ctr[1]:.4f}, {chip_ctr[2]:.4f})")
        if bb:
            print(f"  chip bbox root X [{bb[0]:.4f}, {bb[1]:.4f}]  Y [{bb[2]:.4f}, {bb[3]:.4f}]  "
                  f"Z [{bb[4]:.4f}, {bb[5]:.4f}]  top Z={bb[5]:.4f}")
        print(f"  With _MAG_ROT_WXYZ=-90°Z and _MAG_POS=(tx,ty,tz) world pos ≈")
        print(f"    x = {chip_ctr[1]:.4f} + tx  (root-Y maps to world-X)")
        print(f"    y = {-chip_ctr[0]:.4f} + ty  (root-X maps to world +Y)")
        if bb:
            print(f"    belt top world Z ≈ {bb[5]:.4f} + tz")
        if slot_mouth_x is not None:
            print(f"  slot mouth Y ≈ {slot_mouth_x:.4f} → world Y = {-slot_mouth_x:.4f} + ty")

    # -----------------------------------------------------------------------
    # Materials
    #
    # SteelMaterial (magazine, incl. slot walls): mu_s 0.80 — raised from 0.50 so the PCB
    #   feels real resistance sliding into/through the slot (combine mode is "multiply" with
    #   the PCB's own 0.7/0.5, so effective friction is still < these face values).
    # RailMaterial (Part_1_7): mu_s 0.10 — smooth rail for PCB to slide along (unchanged).
    # BeltMaterial (Part_1_2): mu_s 0.90 — raised from 0.65, rubber-like grip on conveyor.
    # StandMaterial (frame): mu_s 0.50 (unchanged, decorative contact only).
    #
    # Gripper (carriage) friction is configured in widowx_pcb_env_cfg.py via
    #   _GRIPPER_FINGER_STATIC_FRICTION / _GRIPPER_FINGER_DYNAMIC_FRICTION
    #   (event: randomize_rigid_body_material on robot gripper_left/right).
    #
    # Physics Hz: set sim.dt = 1/480 (≈0.00208 s) or 1/240 (≈0.00417 s) in
    #   _WidowXPcbEnvCfgBase.__post_init__ / sim config.
    # -----------------------------------------------------------------------

    steel_mat = MATERIAL_TEMPLATE.format(
        name="SteelMaterial", root=root_path,
        r=0.55, g=0.57, b=0.60,
        roughness=0.30, metallic=0.88,
        static_friction=0.80, dynamic_friction=0.60, restitution=0.01,
    )
    belt_mat = MATERIAL_TEMPLATE.format(
        name="BeltMaterial", root=root_path,
        r=0.10, g=0.10, b=0.11,
        roughness=0.88, metallic=0.0,
        static_friction=0.90, dynamic_friction=0.70, restitution=0.01,
    )
    rail_mat = MATERIAL_TEMPLATE.format(
        name="RailMaterial", root=root_path,
        r=0.32, g=0.35, b=0.40,
        roughness=0.40, metallic=0.12,
        static_friction=0.10, dynamic_friction=0.08, restitution=0.01,
    )
    stand_mat = MATERIAL_TEMPLATE.format(
        name="StandMaterial", root=root_path,
        r=0.44, g=0.46, b=0.50,
        roughness=0.35, metallic=0.82,
        static_friction=0.50, dynamic_friction=0.38, restitution=0.01,
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
{belt_mat}
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


def main():
    script_dir = Path(__file__).parent
    urdf_path  = script_dir / "urdf" / "assembly_6.urdf"
    meshes_dir = script_dir / "meshes"
    # Save in env_v6/ (same folder as this script).
    output_usd = script_dir / "pcb_insertion_env.usd"

    if not urdf_path.exists():
        print(f"ERROR: URDF not found: {urdf_path}")
        return
    if not meshes_dir.exists():
        print(f"ERROR: meshes dir not found: {meshes_dir}")
        return

    create_usd(str(urdf_path), str(meshes_dir), str(output_usd))


if __name__ == "__main__":
    main()
