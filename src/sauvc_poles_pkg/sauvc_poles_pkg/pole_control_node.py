#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from custom_msgs.msg import Telemetry, Commands
from geometry_msgs.msg import PoseStamped


class PoleControlNode(Node):
    def __init__(self):
        super().__init__("pole_control_node")

        # State variables
        self.telemetry = None
        self.pole_pose = None

        # Subscribers
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


        # Publishers
        self.cmd_pub = self.create_publisher(
            Commands,
            "/commands",
            10
        )


        # Controls timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Pole control node started")


    # Callbacks
    def telemetry_callback(self, msg):
        self.telemetry = msg

    def pose_callback(self, msg):
        self.pole_pose = msg

        self.get_logger().info(
            f"Pole pose received: "
            f"x={msg.pose.position.x:.2f}, "
            f"y={msg.pose.position.y:.2f}, "
            f"z={msg.pose.position.z:.2f}"
        )




    def control_loop(self):
        # Safety: if we don't have data yet
        if self.telemetry is None or self.pole_pose is None:
            return
        cmd = Commands()

        cmd.arm = False
        cmd.mode = "MANUAL"

        cmd.forward = 1500
        cmd.lateral = 1500
        cmd.thrust = 1500
        cmd.roll = 1500
        cmd.pitch = 1500
        cmd.yaw = 1500

        cmd.servo1 = 1500
        cmd.servo2 = 1500

        self.cmd_pub.publish(cmd)


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