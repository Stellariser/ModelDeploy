import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.patches import ConnectionPatch
import matplotlib.transforms as mtransforms
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import os

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

    # 获取排序后的聚类中心
    sorted_cluster_centers = cluster_centers[sorted_indices]

    return sorted_cluster_centers

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

def compute_distances_withSIFT(image_path1, image_path2):
    # 加载图像
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 提取关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用BFMatcher进行特征点匹配
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)

    # 根据距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取最相似的匹配对
    best_match = matches[0]

    # 提取最相似特征点的坐标和梯度方向
    pt1 = keypoints1[best_match.queryIdx].pt
    pt2 = keypoints2[best_match.trainIdx].pt
    angle1 = keypoints1[best_match.queryIdx].angle
    angle2 = keypoints2[best_match.trainIdx].angle

    pt1_1 = np.array(keypoints1[matches[0].queryIdx].pt)
    pt1_2 = np.array(keypoints1[matches[1].queryIdx].pt)
    pt2_1 = np.array(keypoints2[matches[0].trainIdx].pt)
    pt2_2 = np.array(keypoints2[matches[1].trainIdx].pt)

    angleList = []

    for i in range(0, 100):
        ptx_1 = np.array(keypoints1[matches[i].queryIdx].pt)
        ptx_2 = np.array(keypoints1[matches[i + 1].queryIdx].pt)
        pty_1 = np.array(keypoints2[matches[i].trainIdx].pt)
        pty_2 = np.array(keypoints2[matches[i + 1].trainIdx].pt)
        v1 = ptx_1 - ptx_2
        v2 = pty_1 - pty_2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angle_degrees = np.degrees(angle)
        angle_degrees_c = 180 - angle_degrees
        finalrotate = min(angle_degrees, angle_degrees_c)
        if not math.isnan(finalrotate):
            angleList.append(finalrotate)

    print(angleList)
    filtered_angles, means = filter_angles(angleList)
    print(filtered_angles)
    print(means)

    print(pt1, angle1, pt2, angle2)


    return (pt1, angle1), (pt2, angle2)


def compute_distances(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    img = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Matching algorithm
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    distancey = float(kp1[matches[0].queryIdx].pt[1] - kp2[matches[0].trainIdx].pt[1])
    distancex = float(kp1[matches[0].queryIdx].pt[0] - kp2[matches[0].trainIdx].pt[0])
    anglex = kp1[matches[0].queryIdx].angle
    angley = kp2[matches[0].queryIdx].angle

    pt1_1 = np.array(kp1[matches[0].queryIdx].pt)
    pt1_2 = np.array(kp1[matches[1].queryIdx].pt)
    pt2_1 = np.array(kp2[matches[0].trainIdx].pt)
    pt2_2 = np.array(kp2[matches[1].trainIdx].pt)

    print(pt1_1,pt1_2,pt2_1,pt2_2)

    angleList = []

    for i in range(0,len(matches)-1):
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

    # filtered_angles,means = filter_angles(angleList)
    # print(filter_angles_Sort(angleList))
    # print(filtered_angles)
    # print(means)
    # print(filter_angles_Sort(angleList)[0],"AAAAAA")

    # print(distancex,"看这个",kp1[matches[0].queryIdx].pt[0],kp2[matches[0].trainIdx].pt[0])
    return distancex, distancey,filter_angles_Sort(angleList)[0]

def stitch_images_with_shift(img1_path, img2_path, output_path):


    x_shift, y_shift = compute_distances(img2_path, img1_path)
    print(x_shift, y_shift)

    x_shift = abs(int(x_shift))
    y_shift = abs(int(y_shift))

    # Open the input images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Calculate the size of the output image
    width = img1.width + 2*abs(x_shift)
    height = max(img1.height, img2.height + abs(y_shift))

    # Create a new image with the combined dimensions
    stitched_image = Image.new('RGB', (width, height))

    # Paste the first image onto the new image
    stitched_image.paste(img1, (x_shift, 0))

    # Paste the second image with the specified horizontal and vertical shifts
    stitched_image.paste(img2, (x_shift-x_shift, y_shift))

    # Save the stitched image
    stitched_image.save(output_path)

def stitch_images_with_shift_multi(images,output_path):

    shifts = []
    for i in range(0,len(images)-1):
        x_shift, y_shift, angle = compute_distances(images[i], images[i+1])
        x_shift = abs(int(x_shift))
        y_shift = abs(int(y_shift))
        shifts.append((x_shift, y_shift,angle))

    print(shifts,"aaaaaaaaa")

    # Read the first image
    img1 = Image.open(images[0])


    # Calculate the size of the output image
    width = img1.width + 2*sum([shift[0] for shift in shifts])
    height = img1.height + sum([shift[1] for shift in shifts])

    print(width,height)

    # Create a new image with the combined dimensions
    stitched_image = Image.new('RGB', (width, height))

    # Paste the first image onto the new image
    stitched_image.paste(img1, (0+sum([shift[0] for shift in shifts]), 0))

    for i in range(1, len(images)):
        img_path = images[i]
        x_shift, y_shift, angle = shifts[i-1]

        # Open the current image
        img = Image.open(img_path)

        # Calculate the position of the current image
        x_pos = -sum([shift[0] for shift in shifts[:i]])+sum([shift[0] for shift in shifts])
        y_pos = sum([shift[1] for shift in shifts[:i]])

        print(x_pos,y_pos)

        #img = rotateimg(img,int(sum([shift[2] for shift in shifts[:i]])))

        print("转成功了2")

        # Paste the current image with the calculated shifts
        stitched_image.paste(img, (x_pos, y_pos))

    # Save the stitched image
    stitched_image.save(output_path)

img1path = './transformedPic/1.png'
img2path = './transformedPic/2.png'
img3path = './transformedPic/3.png'
img4path = './transformedPic/4.png'
img5pathAngle = './transformedPic/113.jpg'
img6pathAngle = './transformedPic/143.jpg'


if __name__ == '__main__':
    outputpath = './resaaa.png'
    outputpath2 = './resaaab.png'

    imgpath = []
    imgpath2 = []

    imgpath.append(img4path)
    imgpath.append(img3path)
    imgpath.append(img2path)
    imgpath.append(img1path)

    imgpath2.append(img5pathAngle)
    imgpath2.append(img6pathAngle)


    #compute_distances(img2path,img1path)
    #compute_distances_withSIFT(img5pathAngle,img6pathAngle)

    #best_orientation1,best_orientation2 = best_orientation(img5pathAngle,img6pathAngle)
    #print(best_orientation1,best_orientation2)

    #estimate_rotation(img1path,img2path)

    # distancex, distancey, gradient_direction1, gradient_direction2 = compute_distances(img1path,img2path)
    # print(distancex, distancey, gradient_direction1, gradient_direction2)

    stitch_images_with_shift_multi(imgpath,outputpath2)
    #stitch_images_with_shift_and_multi(imgpath,outputpath2)

    #stitch_images_with_shift(img3path, img2path, outputpath)
    print("complete")