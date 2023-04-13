import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import os
import shutil

# ======= constants
VIDEO_FILEPATH = 'calib_input.mp4'
OUTPUT_DIR = 'calib_output'
count = 0

# ======= constants
square_size = 2.49
img_mask = f"./{OUTPUT_DIR}/*.jpeg"
pattern_size = (9, 6)
figsize = (20, 20)

# ===== video to calibrate
video_input = cv2.VideoCapture(VIDEO_FILEPATH)

# ======== if output dir exists, delete it
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# ====== make sure output video directory exists or create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ========== run on all frames
    while True:
        ret, frame = video_input.read()
        if not ret:
            break
        if count % 20 == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'frame_{count}.jpeg'), frame)
        count += 1

def camera_calibration():
    img_names = glob(img_mask)
    num_images = len(img_names)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv2.imread(img_names[0]).shape[:2]

    #plt.figure(figsize=figsize)

    for i, fn in enumerate(img_names):
        print("processing %s... " % fn)
        imgBGR = cv2.imread(fn)

        if imgBGR is None:
            print("Failed to load", fn)
            continue

        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

        assert w == img.shape[1] and h == img.shape[0], f"size: {img.shape[1]} x {img.shape[0]}"
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        # # if you want to better improve the accuracy... cv2.findChessboardCorners already uses cv2.cornerSubPix
        # if found:
        #     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        #     cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if not found:
            print("chessboard not found")
            continue

        if i < 12:
            img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
            #plt.subplot(4, 3, i + 1)
            #plt.imshow(img_w_corners)

        print(f"{fn}... OK")
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)


    #plt.show()
    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    return camera_matrix, dist_coefs 




