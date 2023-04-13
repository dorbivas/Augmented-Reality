# ======= imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======= constants
TEMPLATE_IMAGE_FILEPATH = 'template.jpg'
ANOTHER_IMAGE_FILEPATH = 'another_image.jpg'
VIDEO_FILEPATH = 'input.mp4'
OUTPUT_VIDEO_FILEPATH = 'output.mp4'

# === template image keypoint and descriptors
template = cv2.imread(TEMPLATE_IMAGE_FILEPATH)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
feature_extractor = cv2.SIFT_create()
template_kp, template_des = feature_extractor.detectAndCompute(template_gray, None)

# ===== video input, output and metadata
video_input = cv2.VideoCapture(VIDEO_FILEPATH)
video_output = cv2.VideoWriter(OUTPUT_VIDEO_FILEPATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

# ========== run on all frames
frame_counter = 0
while True:
    # === read frame
    ret, frame = video_input.read()
    if not ret:
        break
    frame_counter += 1

    # === info every 10 frames
    if frame_counter % 10 == 0:
        print('frame number: {}'.format(frame_counter))

    # === convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === find keypoints and descriptors
    feature_extractor = cv2.SIFT_create()
    frame_kp, frame_des = feature_extractor.detectAndCompute(frame_gray, None)

    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(template_des, frame_des, k=2)

    # ====== apply ratio test
    good_and_second_good_matches_list = []
    alfa = 0.000000001
    for m in matches:
        if m[0].distance / (m[1].distance + alfa) < 0.7:
            good_and_second_good_matches_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_matches_list)[:, 0]

    # ======== find homography
    # also in SIFT notebook
    good_kp_template = np.array([template_kp[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([frame_kp[m.trainIdx].pt for m in good_match_arr])
    H, _ = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)

    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    another_image = cv2.imread(ANOTHER_IMAGE_FILEPATH)
    another_image = cv2.resize(another_image, (template.shape[1], template.shape[0]))
    rgb_template_warped = cv2.warpPerspective(another_image, H, (frame.shape[1], frame.shape[0]))
    frame[rgb_template_warped != 0] = rgb_template_warped[rgb_template_warped != 0]

    # =========== plot and save frame
    if frame_counter % 10 == 0:
        plt.imshow(frame[:, :, ::-1])
        plt.show()
    #cv2.imshow(str(frame.shape), frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    video_output.write(frame)

# ======== end all
video_input.release()
video_output.release()
