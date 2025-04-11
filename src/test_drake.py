import time

import cv2
import utils
import omnigibson as og
from keypoint_proposal import KeypointProposer
from og_utils import OGCamera
from camera import RobotCamera
from pynput import keyboard
import numpy as np
from og_env import OG_Env
from fk_solver import FKSolver
from path_spline import PathGenerator
import transform_utils as T

def initialize_cameras(og_env,cam_config):
        """
        ::param poses: list of tuples of (position, orientation) of the cameras
        """
        cams = dict()
        for cam_id in cam_config:
            cam_id = int(cam_id)
            cams[cam_id] = OGCamera(og_env, cam_config[cam_id])
        for _ in range(10): og.sim.render()
        return cams


def init_eef_pose():
    action = np.zeros(12)  
    action[4:7] = [0.0, -0.2, 0.0]  
    action[7:10] = [0.0, 0.0, 0.0] 
    action[10:] = [0.0, 0.0]
    return action


running = True
def on_press(key):
    global running
    if key == keyboard.Key.esc:
        print("ESC pressed, exiting loop...")
        running = False

def main():
    urdf = '/home/zhw/.conda/envs/gibson/lib/python3.10/site-packages/omnigibson/data/assets/models/fetch/fetch.urdf'
    eef_name = 'gripper_link'
    reset_joint_pos = np.array([0.2200, -0.9412, -0.6413, 1.5519,  1.6567, -0.9322,  1.5342,  2.1447])
    dof_idx = [2,4,6,7,8,9,10,11]
    trans_world2robot = np.array([
        [ 9.9966598e-01,  1.1358861e-03, -2.5818890e-02, -7.9906189e-01],
        [-1.1401622e-03,  9.9999934e-01, -1.5090039e-04,  1.8915830e-06],
        [ 2.5818702e-02,  1.8028771e-04,  9.9966663e-01,  6.1851740e-04],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
    ])
    og_joint_name = ['l_wheel_joint', 'r_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'shoulder_pan_joint', 'head_tilt_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 'l_gripper_finger_joint', 'r_gripper_finger_joint']

    path_solver = PathGenerator(urdf, eef_name, reset_joint_pos, dof_idx, trans_world2robot, og_joint_name)
    # start
    start_eef_position = np.array([-0.3180, -0.1281,  0.9060])
    start_eef_ori = np.array([ 0.4699,  0.5198, -0.4747,  0.5326])
    start_eef_pose = np.concatenate([start_eef_position, start_eef_ori])
    keypoints = np.array([[-0.25255756, 0.16720406 ,0.68192867],
                            [-0.3970938 ,0.12950217 ,0.68181886],
                            [ 0.5319556, 0.21342507,0.66371404],
                            [-0.28037292,0.09597062 ,0.70344591],
                            [-0.26905417 ,0.00379744 ,0.71206252],
                            [ 1.10996314 ,-0.12611511,  0.62799189],
                            [-0.23855288 ,-0.23952204 , 0.68195394]])
    keypoints[4] = start_eef_position
    keypoints[4][2] -= 0.05
    keypoint_movable_mask = np.zeros(keypoints.shape[0], dtype=bool)
    keypoint_movable_mask[4] = True
    start_eef_twist = np.zeros(6)
    path_solver.set_start(start_eef_pose, start_eef_twist, keypoints, keypoint_movable_mask)

    # goal
    target_eef_position = np.array([-0.41800341,-0.12799984,0.90599945])
    target_eef_orientation = T.euler2quat(np.array([3.14159265,-3.14159265,-3.14159265]))
    target_eef_pose = np.concatenate([target_eef_position, target_eef_orientation])
    target_eef_twist = np.array([2.51799639 ,0.00758151,  2.50676399, 0.0, 0.0, 0.0])
    path_solver.set_target(target_eef_pose, target_eef_twist)

    # cost
    path_solver.set_costs()

    # step size
    path_solver.set_steps(pos_step_size=0.2, rot_step_size=0.78)
    # optimizer
    path_solver.set_optimizer(time=5.0, tol_ori=0.1, tol_pose=0.1, tol_twist=0.1)



    
    
    


if __name__ == "__main__":
    main()