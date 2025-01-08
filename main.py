import time
import utils
import omnigibson as og



def main():
    config_path = './config.yaml'
    scene_path = './scene.json'
    global_config = utils.get_config(config_path=config_path)
    config = global_config['env']
    config['scene']['scene_file'] = scene_path
    og_env = og.Environment(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim']))

    # Robot
    robot = og_env.robots[0]
    action_dim = robot.action_dim

    og_env.scene.update_initial_state()
    for _ in range(100):
        obs, _, _, _, _ = og_env.step([0.0] * action_dim)
        time.sleep(0.1)
    
    og.shutdown()

    


if __name__ == "__main__":
    main()