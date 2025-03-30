import time

import cv2
import utils
import omnigibson as og
import numpy as np
from omnigibson.utils import control_utils
import omnigibson.lazy as lazy
import transform_utils as T



class FKSolver():
    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
        eef_name,
        reset_joint_pos,
        trans_world2robot
    ):
        # Create robot description, kinematics, and config
        self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.eef_name = eef_name
        self.reset_joint_pos = reset_joint_pos
        self.trans_world2robot = trans_world2robot



    def get_eef_poses(self, joint, in_world=True):

        """
        FK compute the eef pose, orientation in given frame

        Returns:
            (x,y,z) eef position in world frame
            (qx,qy,qz,qw) eef orientation in world frame
        """

        link_name = 'gripper_link'
        pose3_lula = self.kinematics.pose(joint, link_name)

        # get position
        link_position = np.array(pose3_lula.translation)

        # get orientation
        rotation_lula = pose3_lula.rotation
        link_orientation = np.array([rotation_lula.x(), rotation_lula.y(), rotation_lula.z(), rotation_lula.w()])

        # in robot frame
        if not in_world:
            return link_position, link_orientation
        
        # transform to world frame
        trans_robot2eef = T.pose2mat((link_position, link_orientation))
        trans_world2eef = np.dot(self.trans_world2robot, trans_robot2eef)
        pos, orn = T.mat2pose(trans_world2eef)
        return pos, orn
        

    def get_jacobian(self, joint_positions, link_name=None):
        """
        Compute Jacobian matrix at given joint positions for given link

        Args:
            joint_positions (np.array): Joint configuration
            link_name (str): Name of link to compute Jacobian at (default to EEF)
        
        Returns:
            J (np.array): 6 x N Jacobian (linear + angular) [vx,vy,vz,wx,wy,wz]
        """
        if link_name is None:
            link_name = self.eef_name
        J = self.kinematics.jacobian(joint_positions, link_name)
        return np.array(J)  



    def get_link_poses(
        self,
        joint_positions,
        link_names,
    ):
        """
        Given @joint_positions, get poses of the desired links (specified by @link_names)

        Args:
            joint positions (n-array): Joint positions in configuration space
            link_names (list): List of robot link names we want to specify (e.g. "gripper_link")
        
        Returns:
            link_poses (dict): Dictionary mapping each robot link name to its pose
        """
        link_poses = {}
        for link_name in link_names:
            pose3_lula = self.kinematics.pose(joint_positions, link_name)

            # get position
            link_position = pose3_lula.translation

            # get orientation
            rotation_lula = pose3_lula.rotation
            link_orientation = (
                rotation_lula.x(),
                rotation_lula.y(),
                rotation_lula.z(),
                rotation_lula.w(),
            )
            link_poses[link_name] =  (link_position, link_orientation)
        return link_poses