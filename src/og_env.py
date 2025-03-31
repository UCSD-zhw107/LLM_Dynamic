import time
import numpy as np
import os
import datetime
import transform_utils as T
import trimesh
import open3d as o3d
import imageio
import omnigibson as og
from camera import RobotCamera
import utils

class OG_Env():
    def __init__(self, config, scene_file):
        self.video_cache = []
        self.config = config
        self.config['scene']['scene_file'] = scene_file
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        # create omnigibson environment
        self.og_env = og.Environment(dict(scene=self.config['scene'], robots=[self.config['robot']['robot_config']], env=self.config['og_sim']))
        self.og_env.scene.update_initial_state()
        for _ in range(10): og.sim.step()
        # robot vars
        self.robot = self.og_env.robots[0]
        self.__init_robot()

        # initialize cameras
        self.robot_cam = RobotCamera(self.robot,None)

    def __init_robot(self):
        """
        Initialize Robot Variables
        """
        self.dof_idx = np.concatenate([self.robot.trunk_control_idx,
                                  self.robot.arm_control_idx[self.robot.default_arm]])
        self.reset_joint_pos = self.robot.reset_joint_pos[self.dof_idx]

    def get_transform(self):
        """
        Get transformation

        Returns:
            trans_world2robot: 4x4 transformation of robot relative to world
            trans_robot2world: 4x4 transformation of world relative to robot base
        """
        # T of robot in world frame
        self.trans_world2robot = T.pose2mat(self.robot.get_position_orientation())
        # T of world in robot frame
        self.trans_robot2world = T.pose_inv(self.trans_world2robot)

        return self.trans_world2robot, self.trans_robot2world
    

    def get_robot_model(self):
        """
        Find robot model

        Returns:
            robot_description_path: robot description path
            robot_urdf_path: URDF path
            eef_name: name of robot links (arm)
            reset_joint_pose: initial joint pose
            dof_idx: index of contolled dof
        """
        
        robot_description_path=self.robot.robot_arm_descriptor_yamls[self.robot.default_arm]
        robot_urdf_path=self.robot.urdf_path
        eef_name=self.robot.eef_link_names[self.robot.default_arm]
        robot_joint_name = list(self.robot.joints.keys())

        return robot_description_path, robot_urdf_path, eef_name, self.reset_joint_pos, self.dof_idx, robot_joint_name


    def get_robot_obs(self):
        """
        Get robot camera observation

        Returns:
            rgb: RGB image
            points: 3D point in image
            mask: segmants
        """
        cam_obs = self.robot_cam.get_obs()
        rgb = utils.to_numpy(cam_obs['rgb'])
        points = utils.to_numpy(cam_obs['points'])
        mask = utils.to_numpy(cam_obs['seg'])
        
        return rgb, points, mask

    def get_robot_eef(self):
        """
        Robot EEF pose

        Returns:
            eef_position: robot eef pose in world frame
            eef_ori: robot eef orientation (quat xyzw) in world frame
        """
        return self.robot.get_eef_position(), self.robot.get_eef_orientation()
    
    def get_joint_config(self):
        """
        Robot joint config
        """
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        return joint_positions[self.dof_idx], joint_velocities[self.dof_idx]
    
    
    def get_joint_limit(self):
        """
        Robot joint limit
        """
        joint_pose_limit = self.robot.joint_position_limits()
        joint_velocity_limit = self.robot.joint_velocity_limits()

        # joint pose limit
        joint_pose_min = joint_pose_limit[0][self.dof_idx]
        joint_pose_max = joint_pose_limit[1][self.dof_idx]

        # joint velocity limit
        joint_vel_min = joint_velocity_limit[0][self.dof_idx]
        joint_vel_max = joint_velocity_limit[1][self.dof_idx]

        return (joint_pose_min, joint_pose_max), (joint_vel_min, joint_vel_max)
        


    def step_robot(self, control):
        self.og_env.step(control)

    def sim_step(self):
        og.sim.step()

    def terminate(self):
        self.og_env.close()
        og.shutdown()