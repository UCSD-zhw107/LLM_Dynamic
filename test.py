import omnigibson as og


# 创建环境
cfg = {
    'scene': {
        'type': 'InteractiveTraversableScene',
        'scene_model': 'Rs_int',
        'scene_file': './scene.json'
    },
    'robots': [
        {
            "type": "Fetch",
            "name": "skynet_robot",
            "obs_modalities": ["rgb", "depth"],
            "action_modalities": "continuous",
            "action_normalize": False,
            "position": [-0.8, 0.0, 0.0],
            "grasping_mode": "assisted",
            "controller_config": {
                "base": {"name": "DifferentialDriveController"},
                "arm_0": {"name": "OperationalSpaceController", "kp": 250, "kp_limits": [50, 400], "damping_ratio": 0.6},
                "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": [0.0, 1.0], "mode": "smooth"},
                "camera": {"name": "JointController"}
            }
        }
    ],
    'task': {
        "type": "DummyTask",
        "termination_config": {},
        "reward_config": {}
    },
    'env': {
        'physics_frequency': 100,
        'action_frequency': 25,
        'rendering_frequency': 25
    }
}
    
# 初始化环境
env = og.Environment(cfg)

# 启用摄像头控制
#og.sim.enable_viewer_camera_teleoperation()

# 模拟步骤
for _ in range(1000):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())

# 关闭环境
og.shutdown()
