#!/usr/bin/env python3


import rclpy
import cv2
import rclpy.node as Node
import math
import numpy as np
from std_msgs.msg import String


class PolePoseNode(Node):
    def __init__(self):
        super().__init__("pole_pose_node")

    def color_correction(self, frame):
        b, g, r = cv2.split(frame)
    
        self.pub = self.cre
        
        
    
        