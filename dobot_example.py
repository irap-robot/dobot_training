from ImageProc import arucoProjection
from DobotDriver import DobotDriver
import cv2
import numpy as np


dobot_arm = DobotDriver()

print(dobot_arm.get_endEffectorPose())