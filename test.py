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

constraint_generator.generate()