# -*- coding: utf8 -*-


import cv2
import numpy as np
import glob
import os


ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'calibrate')
calibrate_files = []


def preprocess_calibrate():
    """preprocess calibrate."""
    global calibrate_files
    for file in glob.glob(os.path.join(ASSETS_PATH, 'left*.jpg')):
        calibrate_files.append(file)

    for file in glob.glob(os.path.join(ASSETS_PATH, 'right*.jpg')):
        calibrate_files.append(file)


def calibrate():
    """exec calibrate."""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    shape = None

    global calibrate_files
    for fname in calibrate_files:
        img = cv2.imread(fname)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray_image, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

        if shape is None:
            shape = gray_image.shape[::-1]

    # キャリブレーション
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    return mtx, dist


def distortion_correction(original_image, gray_image):
    """distortion correction."""
    mtx, dist = calibrate()

    # 歪み補正
    h, w = gray_image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 歪補正
    dist2 = cv2.undistort(gray_image, mtx, dist, None, newcameramtx)

    # 画像の切り落とし
    x, y, w, h = roi
    return dist2[y:y+h, x:x+w]


def line_processing(gray_image, output_threshold_min=200):
    """dilate and substract."""
    gaussian_blur_image = cv2.GaussianBlur(gray_image.copy(), (7, 7), 1)
    _, threshold = cv2.threshold(gaussian_blur_image.copy(), 125, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(gaussian_blur_image.copy(), kernel, iterations=1)
    diff = cv2.subtract(dilation, gaussian_blur_image.copy())
    inverted_white = 255 - diff
    _, line_threshold = cv2.threshold(inverted_white, output_threshold_min, 255, cv2.THRESH_BINARY)
    return line_threshold


def rect_processing(original_image, line_threshold):
    """rect processing."""
    find_contours_image, contours, hierarchy = cv2.findContours(line_threshold.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    draw_image = cv2.drawContours(line_threshold.copy(), contours, -1, (255, 255, 255), 3)
    th_area = line_threshold.shape[0] * line_threshold.shape[1] / 100
    contours_large = list(filter(lambda c:cv2.contourArea(c) > th_area, contours))

    outputs = []
    rects = []
    approxes = []

    for (i,cnt) in enumerate(contours_large):
        # 面積の計算
        arclen = cv2.arcLength(cnt, True)
        # 周囲長を計算（要は多角形の辺の総和）
        approx = cv2.approxPolyDP(cnt, 0.02 * arclen, True)
        # 小さいやつは除外
        if len(approx) < 4:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if is_video_frame_size(x, y, w, h):
            approxes.append(approx)
            rects.append([x, y, w, h])

            rect = cv2.rectangle(original_image.copy(), (x, y), (x+w, y+h), (255, 255, 255), 2)
            outputs.append(rect)

    return rects, outputs, approxes


def is_video_frame_size(x, y, w, h, threshold=200):
    """check video frame size.

    DVD 68:95
    """
    width = w - x
    height = h - y
    
    # 68:95 = width:height -> height = (95 * width) / 68
    _height = (95 * width) / 68
    loss = height - _height
    if threshold > abs(loss):
        return True

    return False


preprocess_calibrate()
