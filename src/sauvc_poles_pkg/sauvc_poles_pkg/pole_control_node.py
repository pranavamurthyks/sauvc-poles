#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

from custom_msgs.msg import Telemetry, Commands
from geometry_msgs.msg import PoseStamped


class PoleControlNode(Node):
    def __init__(self):
        super().__init__("pole_control_node")

        # -------------------------------
        # State variables
        # -------------------------------
        self.telemetry = None
        self.pole_pose = None

        # -------------------------------
        # Subscribers
        # -------------------------------
        self.telemetry_sub = self.create_subscription(
            Telemetry,
            "/telemetry",
            self.telemetry_callback,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/pole/pose",
            self.pose_callback,
            10
        )

        # -------------------------------
        # Publisher
        # -------------------------------
        self.cmd_pub = self.create_publisher(
            Commands,
            "/commands",
            10
        )

        # -------------------------------
        # Control parameters
        # -------------------------------

        # Outer loop: yaw ANGLE → yaw RATE
        self.Kp_yaw_outer = 2.0        # (rad/s per rad)

        # Inner loop: yaw RATE → PWM
        self.Kp_yaw_inner = 120.0      # (PWM per rad/s)

        self.PWM_NEUTRAL = 1500
        self.PWM_LIMIT = 400

        self.control_dt = 0.1  # seconds

        # -------------------------------
        # Control loop timer
        # -------------------------------
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        self.get_logger().info("Pole control node (cascaded yaw) started")

    # =========================================================
    # Callbacks
    # =========================================================

    def telemetry_callback(self, msg):
        self.telemetry = msg

    def pose_callback(self, msg):
        self.pole_pose = msg

    # =========================================================
    # Control Loop
    # =========================================================

    def control_loop(self):

        # Safety check
        if self.telemetry is None or self.pole_pose is None:
            return

        # -----------------------------------------------------
        # VISION → DESIRED YAW ANGLE
        # -----------------------------------------------------

        self.x_error = self.pole_pose.pose.position.x
        self.z_error = self.pole_pose.pose.position.z

        # Desired yaw angle to face the pole
        self.desired_yaw = math.atan2(self.x_error, self.z_error)

        # -----------------------------------------------------
        # CURRENT YAW FROM IMU
        # -----------------------------------------------------

        self.current_yaw = math.radians(self.telemetry.yaw)  # degrees (assumed)

        # Yaw angle error
        self.yaw_angle_error = self.desired_yaw - self.current_yaw

        # Normalize to [-pi, pi]
        self.yaw_angle_error = math.atan2(
            math.sin(self.yaw_angle_error),
            math.cos(self.yaw_angle_error)
        )

        # -----------------------------------------------------
        # OUTER LOOP (ANGLE → RATE)
        # -----------------------------------------------------

        self.desired_yaw_rate = self.Kp_yaw_outer * self.yaw_angle_error

        # -----------------------------------------------------
        # INNER LOOP (RATE → PWM)
        # -----------------------------------------------------

        self.current_yaw_rate = math.radians(self.telemetry.yawspeed)  # rad/s

        self.yaw_rate_error = self.desired_yaw_rate - self.current_yaw_rate

        self.yaw_pwm = (
            self.PWM_NEUTRAL
            + int(self.Kp_yaw_inner * self.yaw_rate_error)
        )

        # Clamp PWM
        self.yaw_pwm = max(
            self.PWM_NEUTRAL - self.PWM_LIMIT,
            min(self.yaw_pwm, self.PWM_NEUTRAL + self.PWM_LIMIT)
        )

        # -----------------------------------------------------
        # SEND COMMAND
        # -----------------------------------------------------

        self.cmd = Commands()

        self.cmd.arm = True
        self.cmd.mode = "MANUAL"

        self.cmd.forward = self.PWM_NEUTRAL
        self.cmd.lateral = self.PWM_NEUTRAL
        self.cmd.thrust = self.PWM_NEUTRAL
        self.cmd.roll = self.PWM_NEUTRAL
        self.cmd.pitch = self.PWM_NEUTRAL
        self.cmd.yaw = self.yaw_pwm

        self.cmd.servo1 = self.PWM_NEUTRAL
        self.cmd.servo2 = self.PWM_NEUTRAL

        self.cmd_pub.publish(self.cmd)


# =============================================================
# Main
# =============================================================

def main(args=None):
    rclpy.init(args=args)
    node = PoleControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()