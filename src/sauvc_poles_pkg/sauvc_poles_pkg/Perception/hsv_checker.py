# import cv2
# import numpy as np

# VIDEO_PATH = "dataset.mp4"

# def normalize_frame(frame):
#     b, g, r = cv2.split(frame)
#     b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
#     g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
#     r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
#     return cv2.merge((b, g, r))

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         frame_norm, hsv = param

#         b, g, r = frame_norm[y, x]
#         h, s, v = hsv[y, x]

#         print(f"Clicked pixel at (x={x}, y={y})")
#         print(f"BGR (normalized) = ({b}, {g}, {r})")
#         print(f"HSV (normalized) = ({h}, {s}, {v})")
#         print("-" * 40)

# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print("Error opening video")
#     exit()

# cv2.namedWindow("Normalized Frame")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_norm = normalize_frame(frame)
#     hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)

#     cv2.setMouseCallback(
#         "Normalized Frame",
#         mouse_callback,
#         param=(frame_norm, hsv)
#     )

#     cv2.imshow("Normalized Frame", frame_norm)

#     key = cv2.waitKey(0) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




















# import cv2
# import numpy as np

# IMAGE_PATH = "red_pole.png"   # change if needed

# def normalize_frame(frame):
#     b, g, r = cv2.split(frame)
#     b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
#     g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
#     r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
#     return cv2.merge((b, g, r))

# def apply_clahe(frame):
#     frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(frame_lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     frame_lab = cv2.merge((l, a, b))
#     frame_bgr = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)
#     return frame_bgr


# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         frame_norm, hsv = param

#         b, g, r = frame_norm[y, x]
#         h, s, v = hsv[y, x]

#         print(f"Clicked pixel at (x={x}, y={y})")
#         print(f"BGR (normalized) = ({b}, {g}, {r})")
#         print(f"HSV (normalized) = ({h}, {s}, {v})")
#         print("-" * 40)

# # ---- LOAD IMAGE ----
# frame = cv2.imread(IMAGE_PATH)
# if frame is None:
#     print("Error loading image")
#     exit()

# frame_norm = normalize_frame(frame)
# frame_norm_clahe = apply_clahe(frame_norm)
# hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)

# cv2.namedWindow("Normalized Image")
# cv2.setMouseCallback(
#     "Normalized Image",
#     mouse_callback,
#     param=(frame_norm_clahe, hsv)
# )

# # ---- DISPLAY LOOP ----
# while True:
#     cv2.imshow("Normalized Image", frame_norm)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cv2.destroyAllWindows()
















import cv2
import numpy as np
import os

IMAGE_DIR = "./poles_dataset/train/images"   # <-- folder path

def normalize_frame(frame):
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

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_disp, hsv = param

        b, g, r = frame_disp[y, x]
        h, s, v = hsv[y, x]

        print(f"Clicked pixel at (x={x}, y={y})")
        print(f"BGR (displayed) = ({b}, {g}, {r})")
        print(f"HSV (displayed) = ({h}, {s}, {v})")
        print("-" * 40)

# -------- LOAD IMAGE LIST --------
images = []
for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        images.append(os.path.join(IMAGE_DIR, file))

if len(images) == 0:
    print("No images found")
    exit()

cv2.namedWindow("Processed Image")

# -------- IMAGE LOOP --------
for image_path in images:
    frame = cv2.imread(image_path)
    if frame is None:
        continue

    frame_norm = normalize_frame(frame)
    frame_disp = apply_clahe(frame_norm)
    hsv = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2HSV)

    cv2.setMouseCallback(
        "Processed Image",
        mouse_callback,
        param=(frame_disp, hsv)
    )

    print(f"\nViewing: {image_path}")

    while True:
        cv2.imshow("Processed Image", frame_disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()
        else:
            break   # go to next image

cv2.destroyAllWindows()