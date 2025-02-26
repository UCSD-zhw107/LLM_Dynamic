import time

import cv2
import utils
'''import omnigibson as og
from keypoint_proposal import KeypointProposer
from og_utils import OGCamera
from camera import RobotCamera'''
from constraint_generator import ConstraintGenerator



config_path = './config.yaml'
global_config = utils.get_config(config_path=config_path)
constraint_generator = ConstraintGenerator(global_config['constraint_generator'])

constraint_generator.generate(instruction='throw pen into the trash bin in front of the robot and table, you may use keypoint[4] for pen and keypoint[5] for trash bin')