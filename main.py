import time
import utils
import omnigibson as og
from keypoint_proposal import KeypointProposer
from og_utils import OGCamera


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


def main():
    config_path = './config.yaml'
    scene_path = './scene.json'
    global_config = utils.get_config(config_path=config_path)
    config = global_config['env']
    config['scene']['scene_file'] = scene_path
    og_env = og.Environment(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim']))
    og_env.scene.update_initial_state()
    for _ in range(10): og.sim.step()

    # Robot
    robot = og_env.robots[0]
    action_dim = robot.action_dim

    # Camera
    cams = initialize_cameras(og_env, config['camera'])
    
    # Perform Task
    og_env.reset()
    for _ in range(100):
        #obs, _, _, _, _ = og_env.step([0.0] * action_dim)
        og.sim.step()
        time.sleep(0.1)
    
    og.shutdown()

    


if __name__ == "__main__":
    main()