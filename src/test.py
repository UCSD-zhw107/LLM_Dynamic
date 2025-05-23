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
from try_drake import compute_fk, compute_fk_autodff, compute_fk_expression

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
    config_path = './config/config.yaml'
    scene_path = './config/hit_apple/scene.json'
    global_config = utils.get_config(config_path=config_path)
    env_config = global_config['env']
    og_env = OG_Env(env_config, scene_path)


    # Listen Keyboard
    key_listener = keyboard.Listener(on_press=on_press)
    key_listener.start()

    
    description, urdf, eef_name, reset_joint_pos, dof_idx, robot_joint_name = og_env.get_robot_model()
    trans_world2robot, trans_robot2world = og_env.get_transform()
    fk_solver = FKSolver(description, urdf, eef_name, reset_joint_pos,trans_world2robot)
    pos,orn = fk_solver.get_eef_poses(reset_joint_pos)
    jacobian = fk_solver.get_jacobian(reset_joint_pos)
    print(jacobian)
    #compute_fk_expression(urdf,reset_joint_pos, trans_world2robot, eef_name,dof_idx, robot_joint_name)
    compute_fk(urdf,reset_joint_pos, trans_world2robot, eef_name,dof_idx, robot_joint_name)
    eef_p,eef_ori= og_env.get_robot_eef()
    print(pos)
    print(eef_p)
    print(eef_ori)
    print("YES++++++++++++++++")
    
    # Perform Task
    while running:
        #obs, _, _, _, _ = og_env.step([0.0] * action_dim)
        og_env.sim_step()
        time.sleep(0.1)

    if key_listener.is_alive():
        key_listener.stop()
        key_listener.join()

    og_env.terminate()

    


if __name__ == "__main__":
    main()