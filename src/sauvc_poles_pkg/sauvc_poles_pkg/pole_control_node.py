#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

from custom_msgs.msg import Telemetry, Commands
from geometry_msgs.msg import PoseStamped


# STAGES 

# Stage 0 (ZERO) → Align depth to the pole in ALT_HOLD mode using y from tvec by having y_tvec < thesh
# Stage 1 (FAR) → Move with cascaded pid of yaw and surge by detecting bbox
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
        self.yaw_threshold = 0.08 # 5 degrees

        # Variable initialization
        self.stage = "NEAR"


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


        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)


    def telemetry_callback(self, msg):
        self.arm = msg.arm
        self.yaw = msg.yaw
        
    
    def pose_callback(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z
        self.frame = msg.header.frame_id
        self.t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


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
        # yaw pid
        self.arm = True
        self.yaw_error = math.atan2(self.x , self.z) # Calculating yaw error by tan-1(x/z)
        if (self.yaw_error > 0.08):
            self.yaw_cmd = 1500 + (self.kp_yaw * self.yaw_error)
            self.yaw_cmd = self.pwm_clamp(self.yaw_cmd) 
            self.cmd.yaw = self.yaw_cmd
        
        else:
            self.surge_error = self.z
            self.surge_cmd = 1500 + self.kp_surge * self.surge_error
            self.surge_cmd = self.pwm_clamp(self.surge_cmd)
            self.cmd.forward = self.surge_cmd

        



    def pwm_clamp(self, pwm):
        return min(1900, max(1100, pwm))



def main(args = None):
    rclpy.init(args=args)
    pose_control_node = PoseControlNode()
    rclpy.spin(pose_control_node)
    pose_control_node.destroy_node()
    rclpy.shutdown()