#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
from geometry_msgs.msg import PoseStamped


class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__("aruco_pose_node")

        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, "/pole/pose", 10)
        
        # Aruco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_param = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_param)


        
        # Assuming 1280x720 
        self.img_w = 1280
        self.img_h = 720

        self.FOV_H_deg = 110
        self.FOV_H_rad = math.radians(self.FOV_H_deg)
        self.FOV_V_deg = 70
        self.FOV_V_rad = math.radians(self.FOV_V_deg)

        cx = self.img_w / 2 # In a ideal camera cx is img_w/2
        cy = self.img_h / 2

        fx = (self.img_w / 2) / math.tan(self.FOV_H_rad / 2)
        fy = (self.img_h / 2) / math.tan(self.FOV_V_rad / 2)
    

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)


        self.marker_size = 0.18 # In metres

        self.object_points = np.array([
            [-self.marker_size / 2, self.marker_size / 2, 0],
            [self.marker_size / 2, self.marker_size / 2, 0],
            [self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0]
        ])

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)


        # Camera
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.get_logger().error("Could Not open webcam")
            return 
        
        # Setting up the image for aruco detection
        # self.img = cv2.imread("/home/pranavamurthy-ks/sauvc_poles_ws/src/sauvc_poles_pkg/sauvc_poles_pkg/aruco_single.webp")
        # self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.timer = self.create_timer(0.033, self.timer_callback)


    
    def timer_callback(self):
        self.ret, self.frame = self.capture.read()
        if not self.ret:
            return

        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Detecting the marker
        self.corners, self.ids, self.rejected = self.aruco_detector.detectMarkers(self.gray)
        if self.ids is not None:
            self.get_logger().info(f"Detected ID's: {self.ids.flatten()}")
        else:
            return
        
        for i in range(len(self.ids)):
            self.img_points = self.corners[i].reshape(4, 2).astype(np.float32)

            self.success, self.rvec, self.tvec = cv2.solvePnP(self.object_points, self.img_points, self.camera_matrix,
                                                              self.dist_coeffs)
            
            if not self.success:
                continue

            self.get_logger().info(f"ID {self.ids[i][0]}")
            self.get_logger().info(f"Translation (X Y Z) meters: {self.tvec.flatten()}")
            self.get_logger().info(f"Rotation vector: {self.rvec.flatten()}")

            cv2.drawFrameAxes(self.frame, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec, 0.1)

            # Draw detected marker
            cv2.polylines(
                self.frame,
                [self.img_points.astype(int)],
                True,
                (0, 255, 0),
                2
            )

        cv2.imshow("ArUco Pose Estimation", self.frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Node kill requested")
            self.destroy_node()

    def destroy_node(self):
        self.get_logger().info("Cleaning up ArucoPoseNode")
        if self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()
        super().destroy_node()
 



            

        




def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()