import cv2
import numpy as np
 
# 左相机内参
left_camera_matrix = np.array([[484.563100250678, -0.191404488818082, 297.642888780887],
                                         [0., 484.506511531479, 210.416658295139],
                                         [0., 0., 1.]])
 
# 左相机畸变系数:[k1, k2, p1, p2, k3]
left_distortion = np.array([[0.0834561601882441, -0.0863298323807046, -0.00207117624399696, 0.000962107948076844, 0]])
 
# 右相机内参
right_camera_matrix = np.array([[484.491003280667, -0.251991305840247, 314.867723600335],
                                          [0., 484.409155055076, 209.466064560304],
                                            [0., 0., 1.]])
# 右相机畸变系数:[k1, k2, p1, p2, k3]                                          
right_distortion = np.array([[0.0657835231049678, -0.0167993993965074, -0.000271582961064217, 0.000631462312814637, 0]])
 
# om = np.array([-0.00009, 0.02300, -0.00372])
# R = cv2.Rodrigues(om)[0]
 
# 旋转矩阵
R = np.array([[0.999909272535658, 0.000351067737620466, -0.0134656395560715],
                           [ -0.000347413705304350, 0.999999902196570, 0.000273697950231144],
                           [ 0.0134657343256060, -0.000268994970577658, 0.999909296706845]])
# r = np.array([[0.0004], [-0.0138], [-0.0003]])
# r = np.array([[0.0004, -0.0138, -0.0003]])
# R = cv2.Rodrigues(r)[0]
# print(R)
# 平移向量
T = np.array([[-60.1931248809071], [0.146818085626969], [-0.941236114956744]])
 
size = (640, 480)
# size = (360, 640)
 
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
 
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)