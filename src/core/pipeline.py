# src/core/pipeline.py
import cv2
import numpy as np

# config variables
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 100
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 100
HOUGH_MAX_RADIUS = 0  # 0 means OpenCV will pick a max based on image size



def preprocess_plate(image_path):
    """Detects the agar plate in the image and crops/masks everything outside the plate"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[!] Could not read image '{image_path}'")
    img_small = cv2.resize(img, (640, 640))  # resize for YOLO if needed
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # detect circles (agar plates)
    max_radius = int(gray.shape[0] // 1.5) if HOUGH_MAX_RADIUS == 0 else HOUGH_MAX_RADIUS
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=max_radius
    )

    mask = np.zeros_like(gray, dtype=np.uint8)  # black mask
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]  # take the first detected plate
        cv2.circle(mask, (x, y), r, 255, -1)  # white inside plate

    # apply mask to keep plate only
    plate_only = cv2.bitwise_and(img_small, img_small, mask=mask)
    return plate_only
