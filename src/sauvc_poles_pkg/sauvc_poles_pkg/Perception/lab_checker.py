import cv2
import numpy as np

VIDEO_PATH = "dataset.mp4"

def normalize_frame(frame):
    b, g, r = cv2.split(frame)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge((b, g, r))


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_norm, lab = param

        b, g, r = frame_norm[y, x]
        l, a, b_lab = lab[y, x]

        print(f"Clicked pixel at (x={x}, y={y})")
        print(f"BGR (normalized) = ({b}, {g}, {r})")
        print(f"LAB (normalized) = (L={l}, a={a}, b={b_lab})")
        print("-" * 40)


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video")
    exit()

cv2.namedWindow("Normalized Frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_norm = normalize_frame(frame)
    lab = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2LAB)

    cv2.setMouseCallback(
        "Normalized Frame",
        mouse_callback,
        param=(frame_norm, lab)
    )

    cv2.imshow("Normalized Frame", frame_norm)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()