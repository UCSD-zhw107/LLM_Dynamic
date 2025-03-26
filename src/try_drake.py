from pathlib import Path
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    Parser,
    RigidTransform,
    RotationMatrix
)
import trimesh
import transform_utils as T
import logging
logging.getLogger("drake").setLevel(logging.ERROR)

def convert_all_stl_to_obj(mesh_dir):
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


'''def map_dof_idx_to_drake_q(dof_idx, joint_names):
    name_to_drake_q_index = {
        'r_wheel_joint': 7,
        'l_wheel_joint': 8,
        'torso_lift_joint': 9,
        'head_pan_joint': 10,
        'head_tilt_joint': 11,
        'shoulder_pan_joint': 12,
        'shoulder_lift_joint': 13,
        'upperarm_roll_joint': 14,
        'elbow_flex_joint': 15,
        'forearm_roll_joint': 16,
        'wrist_flex_joint': 17,
        'wrist_roll_joint': 18,
        'r_gripper_finger_joint': 19,
        'l_gripper_finger_joint': 20,
    }
    return [name_to_drake_q_index[joint_names[i]] for i in dof_idx]'''

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



def compute_fk(urdf_path, joint_positions, T_world_base_4x4, eef_name, dof_idx, og_joint_name):



    # 读取 URDF（支持 STL 替换为 OBJ）
    urdf_path = Path(urdf_path)
    with open(urdf_path, "r") as f:
        urdf_content = f.read()
    urdf_content = urdf_content.replace(".STL", ".obj")

    # 构建 plant
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    model_indices = parser.AddModelsFromString(urdf_content, "urdf")
    model_index = model_indices[0]
    plant.Finalize()

    # 创建 context 并设置 joint 状态
    context = plant.CreateDefaultContext()
    nq = plant.num_positions(model_index)
    q_all = np.zeros(nq)

    drake_joint_name = plant.GetPositionNames(model_index)
    idx = mapidx_og2drake(dof_idx, og_joint_name, drake_joint_name)
    q_all[idx] =joint_positions

    plant.SetPositions(context, model_index, q_all)

    # 设置 base pose（仅适用于 floating base）
    base_body = plant.GetBodyByName("base_link", model_index)
    R = RotationMatrix(T_world_base_4x4[:3, :3])
    p = T_world_base_4x4[:3, 3]
    T_world_base = RigidTransform(R, p)
    plant.SetFreeBodyPose(context, base_body, T_world_base)

    # 计算末端执行器的姿态
    eef_body = plant.GetBodyByName(eef_name, model_index)
    T_world_eef = plant.EvalBodyPoseInWorld(context, eef_body)
    pos,ori = T.mat2pose(T_world_eef.GetAsMatrix4())
    
    #name = plant.GetPositionNames(model_index)
    #print(f"Position #{model_index} -> {name}")

    print(pos)
    return T_world_eef

'''dict_keys(['l_wheel_joint', 8
           'r_wheel_joint', 7
           'torso_lift_joint', 9
           'head_pan_joint', 10
           'shoulder_pan_joint', 12
           'head_tilt_joint', 11
           'shoulder_lift_joint', 13
           'upperarm_roll_joint', 14
           'elbow_flex_joint', 15
           'forearm_roll_joint', 16
           'wrist_flex_joint', 17
           'wrist_roll_joint', 18
           'l_gripper_finger_joint', 19
           'r_gripper_finger_joint'])20
'''


'''['base_link_qw', 
 'base_link_qx', 
 'base_link_qy', 
 'base_link_qz', 
 'base_link_x', 
 'base_link_y', 
 'base_link_z', 
 'r_wheel_joint_q', 
 'l_wheel_joint_q', 
 'torso_lift_joint_x', 
 'head_pan_joint_q', 
 'head_tilt_joint_q', 
 'shoulder_pan_joint_q', 
 'shoulder_lift_joint_q', 
 'upperarm_roll_joint_q', 
 'elbow_flex_joint_q', 
 'forearm_roll_joint_q', 
 'wrist_flex_joint_q', 
 'wrist_roll_joint_q', 
 'r_gripper_finger_joint_x', 
 'l_gripper_finger_joint_x']'''