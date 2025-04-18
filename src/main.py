import time

import cv2
import utils
import omnigibson as og
from keypoint_proposal import KeypointProposer
from og_utils import OGCamera
from camera import RobotCamera
from pynput import keyboard
import numpy as np

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
    env_config['scene']['scene_file'] = scene_path
    og_env = og.Environment(dict(scene=env_config['scene'], robots=[env_config['robot']['robot_config']], env=env_config['og_sim']))
    og_env.scene.update_initial_state()
    for _ in range(10): og.sim.step()





    # Listen Keyboard
    key_listener = keyboard.Listener(on_press=on_press)
    key_listener.start()

    # Set Initial EEF position
    init_eef = init_eef_pose()
    for _ in range(20):
        og_env.step(init_eef)

    # Robot
    robot = og_env.robots[0]

    # Camera
    #cams = initialize_cameras(og_env, config['camera'])
    cam = RobotCamera(robot, None)
    cam_obs = cam.get_obs()
    rgb = utils.to_numpy(cam_obs['rgb'])
    points = utils.to_numpy(cam_obs['points'])
    mask = utils.to_numpy(cam_obs['seg'])

    # KeyPoints
    keypoint_config = global_config['keypoint_proposer']
    keypoint_proposer = KeypointProposer(keypoint_config)
    keypoints, projected_img = keypoint_proposer.get_keypoints(rgb, points, mask)
    cv2.imshow('img', projected_img[..., ::-1])
    cv2.waitKey(0)
    print('showing image, click on the window and press "ESC" to close and continue')
    cv2.destroyAllWindows()
    
    # Write initial keypoint
    with open('initial_keypoint.txt', 'w') as file:
         file.writelines(f'Keypoint: {keypoints}\n')
         file.writelines(f'EEF Pose: {robot.get_eef_position()}\n')
         file.writelines(f'EEF Orientation xyzw: {robot.get_eef_orientation()}\n')

    
    # Perform Task
    while running:
        #obs, _, _, _, _ = og_env.step([0.0] * action_dim)
        og.sim.step()
        time.sleep(0.1)

    if key_listener.is_alive():
        key_listener.stop()
        key_listener.join()

    og_env.close()
    og.shutdown()

    


if __name__ == "__main__":
    main()