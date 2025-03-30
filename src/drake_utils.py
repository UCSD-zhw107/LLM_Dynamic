from pathlib import Path

import trimesh


def convert_stl2obj(mesh_dir):
    """
    Convert .STL file into .obj file
    """
    mesh_dir = Path(mesh_dir)
    if not mesh_dir.is_dir():
        raise ValueError(f"{mesh_dir} not valid path")
    
    stl_files = list(mesh_dir.glob("*.[sS][tT][lL]"))
    if not stl_files:
        print("Can't find .STL file")
        return
    
    success_count = 0
    for stl_file in stl_files:
        obj_file = stl_file.with_suffix(".obj")
        if obj_file.exists():
            continue
        
        try:
            print(f"Exporting: {stl_file.name} to {obj_file.name}")
            mesh = trimesh.load(stl_file, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                mesh.export(obj_file)
                success_count += 1
            else:
                print(f"Skipping: {stl_file.name} not legal mesh")
        except Exception as e:
            print(f"Exporting failed: {stl_file.name} - Error: {str(e)}")
    
    print(f"Exporting Completer: Success {success_count}/{len(stl_files)}")



def mapidx_og2drake(dof_idx, og_joint_names, drake_joint_names):
    """
    Map OG dof_idx to Drake's joint indices based on joint name matching.
    
    Args:
        dof_idx (list[int]): Index list into og_joint_names (e.g., [2, 4, 5])
        og_joint_names (list[str]): All OG joint names (e.g., ['elbow_flex_joint', ...])
        drake_q_names (list[str]): Drake position names from plant.GetPositionNames(model_index)

    Returns:
        list[int]: List of indices in Drake's q_all corresponding to OG's dof_idx
    """
    # mapping table
    drake_name_map = {}
    for i, full_name in enumerate(drake_joint_names):
        for suffix in ["_q", "_x"]:  
            if full_name.endswith(suffix):
                joint_name = full_name.split("/")[-1].replace(suffix, "")
                
                parts = joint_name.split("_")
                if len(parts) > 2:
                    joint_core = "_".join(parts[-3:])  
                else:
                    joint_core = joint_name
                drake_name_map[joint_core] = i

    # map og joint idx into drake joint idx
    mapped_idx = []
    for i in dof_idx:
        joint_name = og_joint_names[i]
        if joint_name not in drake_name_map:
            raise ValueError(f"[ERROR] Joint '{joint_name}' not found in Drake's q names.")
        mapped_idx.append(drake_name_map[joint_name])
    return mapped_idx