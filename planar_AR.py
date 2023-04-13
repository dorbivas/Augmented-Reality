# ======= imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mesh_renderer
import glob
import camera_calibration




# ======= constants
TEMPLATE_IMAGE_FILEPATH = 'template.jpg'
VIDEO_FILEPATH = 'input.mp4'
OUTPUT_VIDEO_FILEPATH = 'output_AR.mp4'
VIDEO_SIZE = (1280, 720)

# ======== camera calibration
K, dist_coeffs = camera_calibration.camera_calibration()

# === template image keypoint and descriptors
template = cv2.imread(TEMPLATE_IMAGE_FILEPATH)
TEMPLATE_SIZE_IN_CM = (21.1, 17)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
feature_extractor = cv2.SIFT_create()
template_kp, template_des = feature_extractor.detectAndCompute(template_gray, None)

# ===== video input, output and metadata
video_input = cv2.VideoCapture(VIDEO_FILEPATH)
video_output = cv2.VideoWriter(OUTPUT_VIDEO_FILEPATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, VIDEO_SIZE)

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
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)

    # ++++++++ take subset of keypoints that obey homography (both frame and reference)
    # this is at most 3 lines- 2 of which are really the same
    # HINT: the function from above should give you this almost completely
    indices = np.where(masked.ravel() > 0)[0]
    good_kp_template = good_kp_template[indices, :]
    good_kp_frame = good_kp_frame[indices, :]

    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K
    # - camera dist_coeffs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows
    xyz = np.array([good_kp_template[:, 0] * TEMPLATE_SIZE_IN_CM[0] / frame.shape[1],
                    good_kp_template[:, 1] * TEMPLATE_SIZE_IN_CM[1] / frame.shape[0],
                    np.zeros(good_kp_template.shape[0])]).T
    ret, r_vec, t_vec = cv2.solvePnP(xyz, good_kp_frame, K, dist_coeffs)

    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
    obj_3d = mesh_renderer.MeshRenderer(K, VIDEO_SIZE[0], VIDEO_SIZE[1], "cat/cat.obj")
    obj_3d.draw(frame, r_vec, t_vec)

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
