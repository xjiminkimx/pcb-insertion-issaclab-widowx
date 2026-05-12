#!/usr/bin/env python3
"""
Enhanced PCB Insertion Environment USD converter.
Converts STL meshes to USD geometry and creates a complete assembly.
"""

import os
from pathlib import Path
from xml.etree import ElementTree as ET
import struct
import math


def parse_urdf_mesh_location(urdf_path):
    """Extract mesh locations and link information from URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    meshes = {}
    for link in root.findall(".//link"):
        link_name = link.get("name")
        visual = link.find("visual")
        if visual is not None:
            geometry = visual.find("geometry/mesh")
            if geometry is not None:
                mesh_file = geometry.get("filename")
                origin = visual.find("origin")
                pos = (0, 0, 0)
                rot = (0, 0, 0)
                if origin is not None:
                    xyz_str = origin.get("xyz", "0 0 0")
                    rpy_str = origin.get("rpy", "0 0 0")
                    pos = tuple(map(float, xyz_str.split()))
                    rot = tuple(map(float, rpy_str.split()))
                
                meshes[link_name] = {
                    "file": mesh_file,
                    "position": pos,
                    "rotation": rot,
                }
    
    return meshes


def read_stl_ascii(filename):
    """Read ASCII STL file and return vertices."""
    vertices = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('vertex'):
                    parts = line.split()
                    if len(parts) == 4:
                        vertex = tuple(map(float, parts[1:4]))
                        vertices.append(vertex)
    except:
        pass
    return vertices


def read_stl_binary(filename):
    """Read binary STL file and return vertices."""
    vertices = []
    try:
        with open(filename, 'rb') as f:
            f.read(80)  # Skip header
            num_triangles = struct.unpack('I', f.read(4))[0]
            for _ in range(num_triangles):
                f.read(12)  # Skip normal
                v1 = struct.unpack('fff', f.read(12))
                v2 = struct.unpack('fff', f.read(12))
                v3 = struct.unpack('fff', f.read(12))
                vertices.extend([v1, v2, v3])
                f.read(2)  # Skip attribute
    except:
        pass
    return vertices


def rpy_to_quat(rpy):
    """Convert RPY (roll, pitch, yaw) to quaternion (x, y, z, w)."""
    roll, pitch, yaw = rpy
    
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (x, y, z, w)


def get_mesh_path(urdf_meshes_dir, mesh_ref):
    """Convert package:// reference to actual file path."""
    if "package://" in mesh_ref:
        mesh_file = mesh_ref.split("meshes/")[-1]
        return os.path.join(urdf_meshes_dir, mesh_file)
    return mesh_ref


def create_mesh_geometry_usd(vertices, name, position, rotation_quat, material_path):
    """Create USD geometry from vertices."""
    if not vertices:
        return ""
    
    x, y, z = position
    rx, ry, rz, rw = rotation_quat
    
    # Calculate bounding box for size estimate
    if len(vertices) > 0:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        
        # Create vertex indices (triangles)
        point_count = len(vertices)
        face_vertex_counts = [3] * (point_count // 3)
        face_vertex_indices = list(range(point_count))
        
        points_str = ", ".join([f"({v[0]}, {v[1]}, {v[2]})" for v in vertices])
        
        usd_str = f'''def Xform "{name}"
{{
    quatf xformOp:orient = ({rx}, {ry}, {rz}, {rw})
    float3 xformOp:translate = ({x}, {y}, {z})
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    
    def Mesh "mesh"
    {{
        uniform bool doubleSided = true
        rel material:binding = <{material_path}>
        
        point3f[] points = [{points_str}]
        int[] faceVertexIndices = {face_vertex_indices}
        int[] faceVertexCounts = {face_vertex_counts}
        
        uniform token purposes = "default"
    }}
}}

'''
        return usd_str
    
    return ""


def create_usd_header():
    """Create USD file header."""
    return '''#usda 1.0
(
    defaultPrim = "World"
    doc = "PCB Insertion Environment for WidowX manipulation"
    metersPerUnit = 1
    timeCodesPerSecond = 24
    upAxis = "Z"
)

'''


def create_material_def(name, color_rgb, roughness=0.5, metallic=0.0):
    """Create a material definition in USD."""
    r, g, b = color_rgb
    return f'''def Material "{name}"
{{
    token inputs:frame:stPrimvarName = "st"
    token outputs:surface.connect = </{name}/Shader.outputs:surface>

    def Shader "Shader"
    {{
        uniform token info:implementationSource = "sourceAsset"
        uniform asset info:sourceAsset = @usdPreviewSurface@
        color3f inputs:diffuseColor = ({r}, {g}, {b})
        float inputs:roughness = {roughness}
        float inputs:metallic = {metallic}
        token outputs:surface
    }}
}}

'''


def create_pcb_insertion_env_usd_v2(urdf_path, meshes_dir, output_usd_path):
    """Create USD file with geometry meshes."""
    
    # Parse URDF
    meshes = parse_urdf_mesh_location(urdf_path)
    
    # Build USD content
    usd_content = create_usd_header()
    
    # Add World scope
    usd_content += '''def Xform "World"
{
    def Scope "Materials"
    {
'''
    
    # Add materials
    usd_content += create_material_def("MagazineMaterial", (0.615686, 0.811765, 0.929412), 0.6, 0.0)
    usd_content += create_material_def("RailMaterial", (0.7, 0.7, 0.7), 0.3, 0.8)
    usd_content += create_material_def("PCBMaterial", (0.2, 0.8, 0.2), 0.4, 0.1)
    
    usd_content += '''    }

    def Xform "PCB_Insertion_Assembly"
    {
'''
    
    # Process key components
    component_count = 0
    
    # Add Magazine (main structure)
    if "magazine" in meshes:
        mag_info = meshes["magazine"]
        mesh_file = get_mesh_path(meshes_dir, mag_info["file"])
        
        if os.path.exists(mesh_file):
            try:
                # Try binary first, then ASCII
                if mesh_file.endswith('.stl'):
                    vertices = read_stl_binary(mesh_file)
                    if not vertices:
                        vertices = read_stl_ascii(mesh_file)
                    
                    if vertices:
                        pos = mag_info["position"]
                        rot_quat = rpy_to_quat(mag_info["rotation"])
                        usd_content += create_mesh_geometry_usd(
                            vertices, 
                            "Magazine",
                            pos,
                            rot_quat,
                            "</World/Materials/MagazineMaterial>"
                        )
                        component_count += 1
            except Exception as e:
                print(f"Warning: Could not process magazine mesh: {e}")
    
    # Add guide rails
    rail_parts = ["part_1_2", "part_1_3", "part_1_8"]
    rail_idx = 0
    for part_name in rail_parts:
        if part_name in meshes:
            rail_info = meshes[part_name]
            mesh_file = get_mesh_path(meshes_dir, rail_info["file"])
            
            if os.path.exists(mesh_file):
                try:
                    if mesh_file.endswith('.stl'):
                        vertices = read_stl_binary(mesh_file)
                        if not vertices:
                            vertices = read_stl_ascii(mesh_file)
                        
                        if vertices:
                            pos = rail_info["position"]
                            rot_quat = rpy_to_quat(rail_info["rotation"])
                            usd_content += create_mesh_geometry_usd(
                                vertices,
                                f"Rail_{rail_idx}",
                                pos,
                                rot_quat,
                                "</World/Materials/RailMaterial>"
                            )
                            rail_idx += 1
                            component_count += 1
                except Exception as e:
                    print(f"Warning: Could not process {part_name} mesh: {e}")
    
    # Add PCB
    if "chip" in meshes:
        chip_info = meshes["chip"]
        mesh_file = get_mesh_path(meshes_dir, chip_info["file"])
        
        if os.path.exists(mesh_file):
            try:
                if mesh_file.endswith('.stl'):
                    vertices = read_stl_binary(mesh_file)
                    if not vertices:
                        vertices = read_stl_ascii(mesh_file)
                    
                    if vertices:
                        pos = chip_info["position"]
                        rot_quat = rpy_to_quat(chip_info["rotation"])
                        usd_content += create_mesh_geometry_usd(
                            vertices,
                            "PCB",
                            pos,
                            rot_quat,
                            "</World/Materials/PCBMaterial>"
                        )
                        component_count += 1
            except Exception as e:
                print(f"Warning: Could not process PCB mesh: {e}")
    
    usd_content += '''    }
}
'''
    
    # Write USD file
    with open(output_usd_path, 'w') as f:
        f.write(usd_content)
    
    print(f"Created enhanced USD file: {output_usd_path}")
    print(f"Components processed: {component_count}")


def main():
    """Main conversion function."""
    script_dir = Path(__file__).parent
    urdf_path = script_dir / "PCB_Insertion_Env" / "urdf" / "assembly_1.urdf"
    meshes_dir = script_dir / "PCB_Insertion_Env" / "meshes"
    output_usd = script_dir / "pcb_insertion_env.usd"
    
    if not urdf_path.exists():
        print(f"URDF not found: {urdf_path}")
        return
    
    print(f"Converting URDF: {urdf_path}")
    print(f"Meshes directory: {meshes_dir}")
    print(f"Output USD: {output_usd}")
    
    create_pcb_insertion_env_usd_v2(str(urdf_path), str(meshes_dir), str(output_usd))


if __name__ == "__main__":
    main()
