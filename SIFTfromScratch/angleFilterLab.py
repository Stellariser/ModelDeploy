import math
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.patches import ConnectionPatch
import matplotlib.transforms as mtransforms
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
import os
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import time


def filter_angles_Sort(angles, bandwidth=None, bin_seeding=True):
    # 将 angles 转换为列向量

    angles = np.array(angles).reshape(-1, 1)

    if bandwidth is None:
        # 使用默认公式估计带宽
        bandwidth = estimate_bandwidth(angles, quantile=0.2)

    # 使用 MeanShift 算法进行聚类
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
    ms.fit(angles)

    # 获取聚类结果的标签和聚类中心
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    # 按照类的数量对聚类进行排序
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]

    # 根据类的数量对 unique_labels 进行排序
    sorted_labels = unique_labels[np.argsort(counts)[::-1]]

    # 获取数量最多的类的标签值
    most_common_label = sorted_labels[0]

    print(most_common_label)

    return most_common_label


def main_feature_direction(angles, bandwidth=None):
    angles = np.array(angles).reshape(-1, 1)

    if bandwidth is None:
        # 使用默认公式估计带宽
        bandwidth = estimate_bandwidth(angles, quantile=0.2)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(angles)

    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    label_counts = Counter(labels)
    most_common_label, _ = label_counts.most_common(1)[0]

    main_direction = cluster_centers[most_common_label]

    return main_direction


def filter_angles(angles, n_clusters=4):
    angles = np.array(angles).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(angles)
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    filtered_angles = angles[labels == np.argmax(cluster_counts)]
    means = []
    for i in range(n_clusters):
        group = angles[labels == i]
        means.append(np.mean(group))
    return filtered_angles, means


def compute_distances(img1_path, img2_path):


    img1 = cv2.imread(img1_path,cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path,cv2.IMREAD_UNCHANGED)

    start_time = time.time()

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # img = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Matching algorithm
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print("Matches",len(matches))

    distancey = float(kp1[matches[0].queryIdx].pt[1] - kp2[matches[0].trainIdx].pt[1])
    distancex = float(kp1[matches[0].queryIdx].pt[0] - kp2[matches[0].trainIdx].pt[0])

    kpf = (kp1[matches[0].queryIdx].pt[0],kp1[matches[0].queryIdx].pt[1])
    kps = (kp2[matches[0].trainIdx].pt[0],kp2[matches[0].trainIdx].pt[1])

    # pt1_1 = np.array(kp1[matches[0].queryIdx].pt)
    # pt1_2 = np.array(kp1[matches[1].queryIdx].pt)
    # pt2_1 = np.array(kp2[matches[0].trainIdx].pt)
    # pt2_2 = np.array(kp2[matches[1].trainIdx].pt)
    #
    # print(pt1_1,pt1_2,pt2_1,pt2_2)

    angleList = []

    for i in range(0, len(matches)-1):
        for j in range(1, len(matches)):
            if i == j:
                continue
            else:
                ptx_1 = np.array(kp1[matches[i].queryIdx].pt)
                ptx_2 = np.array(kp1[matches[i+1].queryIdx].pt)
                pty_1 = np.array(kp2[matches[i].trainIdx].pt)
                pty_2 = np.array(kp2[matches[i+1].trainIdx].pt)
                v1 = ptx_1 - ptx_2
                v2 = pty_1 - pty_2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angle_degrees = np.degrees(angle)
                angle_degrees_c = 180 - angle_degrees

                # 计算叉积，确定旋转方向
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                finalrotate = min(angle_degrees, angle_degrees_c)
                if cross_product < 0:
                    finalrotate = -finalrotate

                angleList.append(finalrotate)

    # 使用seaborn绘制分布图
    print(angleList)
    sns.set(style="whitegrid")
    sns.histplot(angleList, kde=True, color="blue")

    # 设置标题和轴标签
    plt.title("distribution")
    plt.xlabel("angle")
    plt.ylabel("number")

    plt.show()
    # filtered_angles,means = filter_angles(angleList)
    # print(filter_angles_Sort(angleList))
    # print(filtered_angles)
    # print(means)
    # print(filter_angles_Sort(angleList)[0],"AAAAAA")

    # print(distancex,"看这个",kp1[matches[0].queryIdx].pt[0],kp2[matches[0].trainIdx].pt[0])
    return distancex, distancey, filter_angles_Sort(angleList), kpf, kps

img1path = './bmp/1.png'
img2path = './bmp/2.png'
img3path = './bmp/3.png'
img4path = './bmp/4.png'
img5pathAngle = './transformedPic/113.jpg'
img6pathAngle = './transformedPic/143.jpg'
imgpath = []
imgpath2 = []

imgpath.append(img4path)
imgpath.append(img3path)
imgpath.append(img2path)
imgpath.append(img1path)

imgpath2.append(img5pathAngle)
imgpath2.append(img6pathAngle)

if __name__ == '__main__':
    shifts = []

    imgpath = imgpath
    for i in range(0, len(imgpath) - 1):
        x_shift, y_shift, angle, kpf, kps = compute_distances(imgpath[i], imgpath[i + 1])