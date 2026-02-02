import cv2
import numpy as np


# MINMAX Normalization for enhancing the color of the poles
def color_correction(frame):
    b, g, r = cv2.split(frame)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge((b, g, r))

def apply_clahe(frame):
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(frame_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    frame_lab = cv2.merge((l, a, b))
    frame_bgr = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)
    return frame_bgr


VIDEO_PATH = "dataset.mp4"   
HORIZONTAL_FOV_DEG = 100 


# HSV RANGES FOR RED, YELLOW, AND BLUE
LOWER_RED_1 = np.array([0, 80, 40])
UPPER_RED_1 = np.array([10, 255, 255])

LOWER_RED_2 = np.array([165, 80, 40])
UPPER_RED_2 = np.array([180, 255, 255])

LOWER_YELLOW = np.array([17, 100, 100])
UPPER_YELLOW = np.array([45, 255, 255])

# SAUVC POOL

# LOWER_RED_1 = np.array([0, 100, 30])
# UPPER_RED_1 = np.array([10, 255, 255])

# LOWER_RED_2 = np.array([140, 100, 30])
# UPPER_RED_2 = np.array([180, 255, 255])

# LOWER_YELLOW = np.array([17, 40, 40])    
# UPPER_YELLOW = np.array([80, 255, 255])

LOWER_BLUE = np.array([90, 150, 100])
UPPER_BLUE = np.array([105, 255, 255])


capture = cv2.VideoCapture("dataset.mp4")
if not capture.isOpened():
    print("Error in opening the video")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow("Original Frame", frame)

    # Color correction of the frame by MINMAX normalization
    frame_normalized = color_correction(frame)
    cv2.imshow("Frame after color correction", frame_normalized)

    frame_normalized_clahe = apply_clahe(frame_normalized)
    cv2.imshow("Frame after color correction and CLAHE", frame_normalized_clahe)

    # Applying median blur to reduce noise
    frame_normalized_clahe_blur = cv2.medianBlur(frame_normalized_clahe, 3)

    hsv_frame = cv2.cvtColor(frame_normalized_clahe_blur, cv2.COLOR_BGR2HSV)


    # RED MASK
    mask_1 = cv2.inRange(hsv_frame, LOWER_RED_1, UPPER_RED_1)
    mask_2 = cv2.inRange(hsv_frame, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    # Morphology to smoothen the mask
    kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT ,(3, 25))
    mask_red = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_red)
    cv2.imshow("Red Mask", mask_red)

    # Finding all the contours of the red mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_red, y_red, w_red, h_red = 0, 0, 0, 0
    frame_bbox = frame_normalized_clahe_blur.copy()
    img_h, img_w = frame_bbox.shape[:2]
    pole_candidates_red = []

    for contour in contours_red:
        contour_area = cv2.contourArea(contour)
        if contour_area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = h / w if w != 0 else 0
        if aspect_ratio < 3:
            continue  # not pole like

        # If it passes all conditions, then it might be a pole
        pole_candidates_red.append((contour, x, y, w, h))

    if len(pole_candidates_red) >0:
        best = max(pole_candidates_red, key=lambda item: item[4])
        contour, x_red, y_red, w_red, h_red = best
        cv2.rectangle(frame_bbox, (x_red, y_red), (x_red + w_red, y_red + h_red), (0, 255, 0), 2)
        cv2.putText(frame_bbox, "RED POLE", (x_red, y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculating the yaw angle
        img_center_x = img_w // 2
        pole_center_red_x = x_red + (w_red // 2)
        x_error_red = pole_center_red_x - img_center_x
        yaw_angle_red = (x_error_red / img_center_x) * (HORIZONTAL_FOV_DEG / 2)
        cv2.line(frame_bbox, (img_center_x, 0), (img_center_x, img_h), (255, 0, 0), 2)
        cv2.circle(frame_bbox, (pole_center_red_x, y_red), 6, (0, 0, 255), -1)
        cv2.putText(frame_bbox, f"Yaw Red: {yaw_angle_red:.2f} deg", (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2)


    # YELLOW MASK
    mask_yellow = cv2.inRange(hsv_frame, LOWER_YELLOW, UPPER_YELLOW)
    kernel_yellow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)) # (3, 75)
    # kernel_yellow_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel_yellow)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel_yellow)
    # mask_yellow = cv2.dilate(mask_yellow, kernel_yellow_2, iterations=1)
    cv2.imshow("Yellow Mask", mask_yellow)

    # Finding all the contours of the yellow mask
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_yellow, y_yellow, w_yellow, h_yellow = 0, 0, 0, 0
    pole_candidates_yellow = []

    for contour in contours_yellow:
        contour_area = cv2.contourArea(contour)
        if contour_area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = h / w if w != 0 else 0
        if aspect_ratio < 3:
            continue  # not pole like

        # If it passes all conditions, then it might be a pole
        pole_candidates_yellow.append((contour, x, y, w, h))

    if len(pole_candidates_yellow) >0:
        best = max(pole_candidates_yellow, key=lambda item: item[4])
        contour, x_yellow, y_yellow, w_yellow, h_yellow = best
        cv2.rectangle(frame_bbox, (x_yellow, y_yellow), (x_yellow + w_yellow, y_yellow + h_yellow), (0, 255, 0), 2)
        cv2.putText(frame_bbox, "YELLOW POLE", (x_yellow, y_yellow - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculating the yaw angle
        img_center_x = img_w // 2
        pole_center_yellow_x = x_yellow + (w_yellow // 2)
        x_error_yellow = pole_center_yellow_x - img_center_x
        yaw_angle_yellow = (x_error_yellow / img_center_x) * (HORIZONTAL_FOV_DEG / 2)
        cv2.circle(frame_bbox, (pole_center_yellow_x, y_yellow), 6, (0, 0, 255), -1)
        cv2.putText(frame_bbox, f"Yaw Yellow: {yaw_angle_yellow:.2f} deg", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 255), 2)
    
    cv2.imshow("Bounding box", frame_bbox)


    key = cv2.waitKey(0) & 0xFF   
    if key == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()