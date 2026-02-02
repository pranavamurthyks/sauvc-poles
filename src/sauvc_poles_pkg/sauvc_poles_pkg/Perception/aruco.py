# import cv2
# import numpy

# # Aruco Setup
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# aruco_param = cv2.aruco.DetectorParameters()
# aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)

# capture = cv2.VideoCapture(0)
# if not capture.isOpened():
#     print("Error opening camera")
#     exit()

# while True:
#     ret, frame = capture.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     corners, ids, rejected = aruco_detector.detectMarkers(gray)

#     if ids is not None:
#         cv2.aruco.drawDetectedMarkers(frame, corners, ids)
#         print("Detected IDs:", ids.flatten())

#     cv2.imshow("Aruco Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
import math

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

# =========================
# ARUCO SETUP
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# =========================
# CAMERA INTRINSICS (APPROX)
# =========================
img_w = 1280
img_h = 720

FOV_H_deg = 70  # MacBook webcam approx
FOV_H_rad = math.radians(FOV_H_deg)

cx = img_w / 2
cy = img_h / 2

fx = (img_w / 2) / math.tan(FOV_H_rad / 2)
fy = fx  # assume square pixels

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)

dist_coeffs = np.zeros((5, 1))  # assume no distortion for now

# =========================
# MARKER MODEL (3D POINTS)
# =========================c
marker_length = 0.18  # meters (18 cm)

obj_points = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            img_points = corners[i].reshape(4, 2).astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs
            )

            if not success:
                continue

            # Print pose
            print(f"ID {ids[i][0]}")
            print("Translation (X Y Z) meters:", tvec.flatten())
            print("Rotation vector:", rvec.flatten())
            print("-" * 30)

            # Draw axes
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                0.1
            )

            # Draw detected marker
            cv2.polylines(
                frame,
                [img_points.astype(int)],
                True,
                (0, 255, 0),
                2
            )

    cv2.imshow("ArUco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()