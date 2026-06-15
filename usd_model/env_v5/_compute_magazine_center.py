#!/usr/bin/env python3
"""One-off: magazine bbox/centre in env-local frame for env_v5 (assembly_1)."""
from pathlib import Path

import convert_to_usd as c

script_dir = Path(__file__).parent
urdf_path = script_dir / "urdf" / "assembly_1.urdf"
meshes_dir = script_dir / "meshes"

links, joints = c.parse_urdf(str(urdf_path))
fk = c.compute_fk(joints, root_link="part_1_1")

mag_points: list = []
for link_name in sorted(c.MAGAZINE_LINKS):
    info = links.get(link_name)
    if info is None or info["mesh"] is None:
        continue
    stl_path = c.get_mesh_path(str(meshes_dir), info["mesh"])
    triangles = c.read_stl_binary(stl_path)
    T_link = fk.get(link_name, c.IDENTITY_T)
    T_visual = c.make_transform(info["visual_pos"], info["visual_rpy"])
    T_world = c.compose_transforms(T_link, T_visual)
    tris = c.transform_triangles(triangles, T_world)
    points, _ = c.triangles_to_arrays(tris)
    mag_points.extend(points)

xs = [p[0] for p in mag_points]
ys = [p[1] for p in mag_points]
zs = [p[2] for p in mag_points]
cx = 0.5 * (min(xs) + max(xs))
cy = 0.5 * (min(ys) + max(ys))
cz = 0.5 * (min(zs) + max(zs))

print("Magazine bbox in USD root frame (part_1_1):")
print(f"  X: [{min(xs):.4f}, {max(xs):.4f}]  extent {max(xs)-min(xs):.4f}")
print(f"  Y: [{min(ys):.4f}, {max(ys):.4f}]  extent {max(ys)-min(ys):.4f}")
print(f"  Z: [{min(zs):.4f}, {max(zs):.4f}]  extent {max(zs)-min(zs):.4f}")
print(f"  centre (root): ({cx:.4f}, {cy:.4f}, {cz:.4f})")

tx, ty, tz = (0.478, 0.140, 0.014)
wx = cy + tx
wy = -cx + ty
wz = cz + tz
print(f"\nMagazine centre env-local (_MAG_POS=({tx},{ty},{tz})):")
print(f"  ({wx:.4f}, {wy:.4f}, {wz:.4f})")
