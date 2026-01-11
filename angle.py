# import cv2
# import numpy as np

# def color_correction(frame):
#     b, g, r = cv2.split(frame)

#     b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
#     g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
#     r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

#     corrected = cv2.merge((b, g, r))
#     return corrected


# def apply_clahe(frame):
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     merged = cv2.merge((l, a, b))
#     enhanced_bgr_image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
#     return enhanced_bgr_image


# VIDEO_PATH = "dataset.mp4"   
# HORIZONTAL_FOV_DEG = 100   


# capture = cv2.VideoCapture(VIDEO_PATH)
# if not capture.isOpened():
#     print("Error opening video")
#     exit()


# while True:
#     ret, frame = capture.read()
#     if not ret:
#         break

#     cv2.imshow("Before color correction", frame)

#     frame_normalized = color_correction(frame)
#     cv2.imshow("After color correction", frame_normalized)

#     hsv_image = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2HSV)
    
#     # Yellow pole
#     lower_yellow = np.array([92, 150, 100])
#     upper_yellow = np.array([98, 255, 190])
#     yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
#     # Cyan/Blue pole
#     lower_cyan = np.array([25, 140, 90])
#     upper_cyan = np.array([40, 220, 225])
#     cyan_mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    
#     # Red pole (two ranges because red wraps)
#     lower_red1 = np.array([0, 90, 90])
#     upper_red1 = np.array([7, 220, 195])
#     red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    
#     lower_red2 = np.array([168, 90, 90])
#     upper_red2 = np.array([178, 220, 195])
#     red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
#     red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
#     # Combine all pole masks
#     poles_mask = cv2.bitwise_or(yellow_mask, cyan_mask)
#     poles_mask = cv2.bitwise_or(poles_mask, red_mask)
    
#     cv2.imshow("Poles Mask", poles_mask)

#     key = cv2.waitKey(0) & 0xFF
#     if key == ord('q'):
#         break


# capture.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

def color_correction(frame):
    b, g, r = cv2.split(frame)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    corrected = cv2.merge((b, g, r))
    return corrected


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    enhanced_bgr_image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr_image


def detect_poles_multistage(frame_normalized):
    # STAGE 1: Edge Detection - Find vertical structures
    gray = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly to connect nearby edge pixels
    kernel_edge = np.ones((3, 1), np.uint8)  # Vertical kernel
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)
    
    # STAGE 2: Color Detection in HSV
    hsv_image = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2HSV)
    
    # Yellow pole
    lower_yellow = np.array([92, 150, 100])
    upper_yellow = np.array([98, 255, 190])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    # Cyan/Blue pole
    lower_cyan = np.array([25, 140, 90])
    upper_cyan = np.array([40, 220, 225])
    cyan_mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    
    # Red pole
    lower_red1 = np.array([0, 90, 90])
    upper_red1 = np.array([7, 220, 195])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    
    lower_red2 = np.array([168, 90, 90])
    upper_red2 = np.array([178, 220, 195])
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Combine color masks
    color_mask = cv2.bitwise_or(yellow_mask, cyan_mask)
    color_mask = cv2.bitwise_or(color_mask, red_mask)
    
    # STAGE 3: LAB Color Space - Better for underwater
    lab_image = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    
    # Poles tend to have high saturation in A and B channels
    a_thresh = cv2.threshold(cv2.absdiff(a, 128), 20, 255, cv2.THRESH_BINARY)[1]
    b_thresh = cv2.threshold(cv2.absdiff(b, 128), 20, 255, cv2.THRESH_BINARY)[1]
    lab_mask = cv2.bitwise_or(a_thresh, b_thresh)
    
    # STAGE 4: Combine all evidence (edges + color + LAB)
    # Voting system: if 2 out of 3 agree, it's likely a pole
    combined = np.zeros_like(edges_dilated)
    combined = cv2.bitwise_or(combined, cv2.bitwise_and(edges_dilated, color_mask))
    combined = cv2.bitwise_or(combined, cv2.bitwise_and(edges_dilated, lab_mask))
    combined = cv2.bitwise_or(combined, cv2.bitwise_and(color_mask, lab_mask))
    
    # STAGE 5: Morphology to clean up and connect segments
    kernel_morph = np.ones((5, 3), np.uint8)  # Vertical kernel
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_morph)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_morph)
    
    return combined


VIDEO_PATH = "dataset.mp4"   
HORIZONTAL_FOV_DEG = 100   

capture = cv2.VideoCapture(VIDEO_PATH)
if not capture.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        break

    cv2.imshow("Before color correction", frame)

    frame_normalized = color_correction(frame)
    cv2.imshow("After color correction", frame_normalized)

    # Multi-stage pole detection - output is the final mask
    poles_mask = detect_poles_multistage(frame_normalized)
    
    cv2.imshow("Poles Mask", poles_mask)



capture.release()
cv2.destroyAllWindows()