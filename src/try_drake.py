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


def build(urdf_path):
    urdf_path = Path(urdf_path)
    
    # 读取原始 URDF 内容
    with open(urdf_path, "r") as f:
        content = f.read()

    # 替换 .STL 为 .obj
    content_new = content.replace(".STL", ".obj")

    # 生成新的 URDF 文件路径
    new_urdf_path = urdf_path.with_name(urdf_path.stem + "_drake.urdf")

    # 写入新的 Drake 兼容 URDF 文件
    with open(new_urdf_path, "w") as f:
        f.write(content_new)


def compute_fk(urdf_path, joint_positions, T_world_base_4x4, eef_name, dof_idx):
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
    q_all[dof_idx] = joint_positions

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
    print(pos)
    return T_world_eef