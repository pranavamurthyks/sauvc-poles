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
import numpy

# Aruco Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_param = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)


img = cv2.imread("aruco.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners, ids, rejected = aruco_detector.detectMarkers(gray)

if ids is not None:
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    print("Detected IDs:", ids.flatten())

cv2.imshow("Aruco Detection", img)

cv2.waitKey(0)

cv2.destroyAllWindows()


