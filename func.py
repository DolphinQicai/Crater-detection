import cv2
import os
import numpy as np
import config1280x480

def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2

def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image

def preprocess(img1, img2):
    if(img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


def stereoMatchSGBM(left_image, right_image, down_scale=False):
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }

    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right



def get_distance(iml, imr):
    iml = undistortion(iml, config1280x480.left_camera_matrix, config1280x480.left_distortion)
    imr = undistortion(imr, config1280x480.right_camera_matrix, config1280x480.right_distortion)
    iml_, imr_ = preprocess(iml, imr)
    iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, config1280x480.left_map1, config1280x480.left_map2, \
                                                    config1280x480.right_map1, config1280x480.right_map2)
    disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, True)
    points_3d = cv2.reprojectImageTo3D(disp, config1280x480.Q)
    for x in range(350, 390):
        for y in range(230, 270):
            if points_3d[y, x, 0] > 0 and points_3d[y, x, 1] > 0:
                min_dis = ( (points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] **2) ** 0.5) / 1000
                return min_dis
    return 10000
