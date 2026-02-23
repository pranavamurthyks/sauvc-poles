#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from collections import deque

from custom_msgs.msg import Telemetry, Commands
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import BoundingBox2D
from std_msgs.msg import Float32MultiArray, Float32


# STAGES 

# Stage 0 (ZERO) → Align depth to the pole in ALT_HOLD mode 
# Stage 1 (FAR) → Match center of bbox and frame and apply cascaded pid of depth, yaw and surge
# Stage 2 (MID) → When the auv is close enuf so that segmentation is possible, get pose from solvepnp where the 4 points from the segmentation of the 
#           actual pole is used instead of yolo and apply same cascaded pid with surge and yaw
# Stage 3 (NEAR) → When the auv is really close where the whole pole is no more fully visible, switch to the idea of centering the segmented pole to the 
#           camera frame instead of pose and move forward


class PoseControlNode(Node):
    def __init__(self):
        super().__init__("pose_control_node")

        # PID VALUES
        self.kp_yaw = 100.0
        self.kp_surge = 200.0
        self.kp_depth = 0.1
        self.kp_lateral = 0.1
        self.yaw_threshold = 0.08 # 5 degrees

        # Variable initialization
        self.stage = "NEAR"

        self.bbox_history = deque(maxlen=5)
        self.tvec_history = deque(maxlen=5)
        self.bbox_center_x = None
        self.bbox_center_y = None
        self.bbox_size_x = None
        self.bbox_size_y = None

        self.frame_center_x = None
        self.frame_center_y = None


        self.cmd = Commands()
        self.cmd.arm = False

        self.cmd.mode = "ALT_HOLD"
        self.cmd.forward = 1500
        self.cmd.lateral = 1500
        self.cmd.thrust = 1500
        self.cmd.yaw = 1500
        self.cmd.pitch = 1500
        self.cmd.roll = 1500
        

        # Publisher
        self.cmd_pub = self.create_publisher(Commands, '/master/commands', 10)

        # Subscribers
        self.telemetry_sub = self.create_subscription(Telemetry, '/master/telemetry', self.telemetry_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/pole/pose', self.pose_callback, 10)
        self.bbox_sub = self.create_subscription(BoundingBox2D, '/pole/bbox', self.bbox_callback, 10)
        self.frame_center_sub = self.create_subscription(Float32MultiArray, '/frame/center', self.frame_center_callback, 10)
        self.stage_sub = self.create_subscription(Float32, '/pole/stage', self.stage_callback, 10)


        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)


    def telemetry_callback(self, msg):
        self.arm = msg.arm
        self.yaw = msg.yaw
        
    
    def pose_callback(self, msg):
        self.tvec_history.append(
            {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            }
        )
        if (len(self.tvec_history) == 5):
            self.x = np.mean([f['x'] for f in self.tvec_history])
            self.y = np.mean([f['y'] for f in self.tvec_history])
            self.z = np.mean([f['z'] for f in self.tvec_history])
        self.frame = msg.header.frame_id
        self.t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


    def bbox_callback(self, msg):
        self.bbox_history.append(
            {
                'cx': msg.center.position.x,
                'cy': msg.center.position.y,
                'w': msg.size_x,
                'h': msg.size_y
            })
        
        if (len(self.bbox_history) == 5):
            self.bbox_center_x = np.mean([f['cx'] for f in self.bbox_history])
            self.bbox_center_y = np.mean([f['cy'] for f in self.bbox_history])
            self.bbox_size_x = np.mean([f['w'] for f in self.bbox_history])
            self.bbox_size_y = np.mean([f['h'] for f in self.bbox_history])

    def frame_center_callback(self, msg):
        self.frame_center_x = msg.data[0]
        self.frame_center_y = msg.data[1]

    
    def stage_callback(self, msg):
        if (msg - 0.0) < 0.5: # Using 0.5 as threshold as float and integer cannot be compared directly
            self.stage = "ZERO"
        elif (msg - 1.0) < 0.5:
            self.stage = "FAR"
        elif (msg - 2.0) < 0.5:
            self.stage = "MID"
        elif (msg - 3.0) < 0.5:
            self.stage = "NEAR"
        else:
            self.stage = None


    def control_loop(self):
        if self.stage == "MID":
            self.control_mid()
        elif self.stage == "FAR":
            self.control_far()
        elif self.stage == "NEAR":
            self.control_near()
        elif self.stage == "ZERO":
            self.control_zero()
        else:
            self.stable_search()
        self.cmd_pub.publish(self.cmd)
    

    def control_mid(self):
        if self.x is None or self.z is None:
            return
        # yaw pid
        self.arm = True
        self.yaw_error = math.atan2(self.x , self.z) # Calculating yaw error by tan-1(x/z)
        if (abs(self.yaw_error) > 0.08): # 0.08 is 5 degrees
            self.yaw_cmd = 1500 + (self.kp_yaw * self.yaw_error)
            self.yaw_cmd = self.pwm_clamp(self.yaw_cmd) 
            self.cmd.yaw = self.yaw_cmd
        
        else:
            self.surge_error = self.z
            self.surge_cmd = 1500 + (self.kp_surge * self.surge_error)
            self.surge_cmd = self.pwm_clamp(self.surge_cmd)
            self.cmd.forward = self.surge_cmd

    

    def control_far(self):
        if self.bbox_center_x is None or self.frame_center_x is None:
            return
        self.arm = True
        self.depth_error = self.frame_center_y - self.bbox_center_y
        if (abs(self.depth_error) > 20):
            self.thrust_cmd = 1500 + (self.depth_error * self.kp_depth)
            self.thrust_cmd = self.pwm_clamp(self.thrust_cmd)
            self.cmd.thrust = self.thrust_cmd
        else:
            self.lateral_error = self.bbox_center_x - self.frame_center_x
            if (abs(self.lateral_error) > 20):
                self.lateral_cmd = 1500 + (self.kp_lateral * self.lateral_error)
                self.lateral_cmd = self.pwm_clamp(self.lateral_cmd)
                self.cmd.lateral = self.lateral_cmd
            else:
                self.cmd.forward = 1600
            

    def pwm_clamp(self, pwm):
        return min(1900, max(1100, pwm))



def main(args = None):
    rclpy.init(args=args)
    pose_control_node = PoseControlNode()
    rclpy.spin(pose_control_node)
    pose_control_node.destroy_node()
    rclpy.shutdown()

if __name__  == "__main__":
    main()