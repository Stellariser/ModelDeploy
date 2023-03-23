import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.patches import ConnectionPatch
import matplotlib.transforms as mtransforms
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import os

def display_custom_points_matplotlib(img1_path, img2_path, coord1, coord2):
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    # 在图像上绘制点
    img1 = cv2.circle(img1, coord1, radius=4, color=(0, 0, 255), thickness=-1)
    img2 = cv2.circle(img2, coord2, radius=4, color=(0, 0, 255), thickness=-1)

    # 将图像从BGR转换为RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 使用matplotlib显示图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax1.set_title("Image 1")
    ax1.axis("off")

    ax2.imshow(img2)
    ax2.set_title("Image 2")
    ax2.axis("off")


    plt.show()

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
        shifts.append((x_shift, y_shift))

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
        x_shift, y_shift = shifts[i-1]

        # Open the current image
        img = Image.open(img_path)

        # Calculate the position of the current image
        x_pos = -sum([shift[0] for shift in shifts[:i]])+sum([shift[0] for shift in shifts])
        y_pos = sum([shift[1] for shift in shifts[:i]])

        print(x_pos,y_pos)

        # Paste the current image with the calculated shifts
        stitched_image.paste(img, (x_pos, y_pos))

    # Save the stitched image
    stitched_image.save(output_path)


def stitch_images_with_shift_and_multi(images, output_path):
    shifts_and_angles = []
    for i in range(0, len(images) - 1):
        x_shift, y_shift, angle = compute_distances(images[i], images[i + 1])
        x_shift = abs(int(x_shift))
        y_shift = abs(int(y_shift))
        shifts_and_angles.append((x_shift, y_shift, angle))

    print(shifts_and_angles)

    img1 = Image.open(images[0])
    width = img1.width + 2 * sum([shift[0] for (shift, _, _) in shifts_and_angles])
    height = img1.height + sum([shift[1] for (shift, _, _) in shifts_and_angles])

    print(width, height)

    stitched_image = Image.new('RGB', (width, height))
    stitched_image.paste(img1, (0 + sum([shift[0] for (shift, _, _) in shifts_and_angles]), 0))

    for i in range(1, len(images)):
        img_path = images[i]
        x_shift, y_shift, angle = shifts_and_angles[i - 1]

        img = Image.open(img_path)
        img_np = np.array(img)

        center = (img_np.shape[1] // 2, img_np.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img_np = cv2.warpAffine(img_np, rotation_matrix, (img_np.shape[1], img_np.shape[0]))
        rotated_img = Image.fromarray(rotated_img_np)

        x_pos = -sum([shift[0] for (shift, _, _) in shifts_and_angles[:i]]) + sum(
            [shift[0] for (shift, _, _) in shifts_and_angles])
        y_pos = sum([shift[1] for (shift, _, _) in shifts_and_angles[:i]])

        print(x_pos, y_pos)
        stitched_image.paste(rotated_img, (x_pos, y_pos))

    stitched_image.save(output_path)

def extract_frames_from_video(video_path, frame_rate):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame if it meets the specified frame rate
        if frame_count % frame_rate == 0:
            frame_img_path = f"frame{frame_count}.png"
            cv2.imwrite(frame_img_path, frame)
            frames.append(frame_img_path)

        frame_count += 1

    # Release the video file
    cap.release()
    return frames

def stitch_frames_from_video(video_path, output_path, frame_rate):
    # Extract frames from the video
    frames = extract_frames_from_video(video_path, frame_rate)

    # Stitch the frames using the existing stitching method
    stitch_images_with_shift_multi(frames, output_path)

    # Clean up the extracted frames
    for frame_img_path in frames:
        os.remove(frame_img_path)

def compute_distancess(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Matching algorithm
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 计算图像的梯度
    sobelx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    sobely1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
    sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)
    sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)

    # 计算梯度方向
    angle1 = cv2.phase(sobelx1, sobely1, angleInDegrees=True)
    angle2 = cv2.phase(sobelx2, sobely2, angleInDegrees=True)

    # 获取特征点的坐标
    x1, y1 = kp1[matches[0].queryIdx].pt
    x2, y2 = kp2[matches[0].trainIdx].pt

    # 获取特征点的梯度方向
    gradient_direction1 = angle1[int(y1), int(x1)]
    gradient_direction2 = angle2[int(y2), int(x2)]

    print("Gradient direction for point 1:", gradient_direction1)
    print("Gradient direction for point 2:", gradient_direction2)

    distancey = float(y1 - y2)
    distancex = float(x1 - x2)

    return distancex, distancey,gradient_direction1,gradient_direction2

def estimate_rotation(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 计算关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 初始化BFMatcher（Brute Force Matcher）
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用Lowe's比率测试（筛选好的匹配点）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 计算旋转角度
        if H is not None:
            _, _, theta = cv2.decomposeHomographyMat(H)        #相机内参矩阵
            for t in theta:
                t = np.rad2deg(t) % 360  # 转换为0-360度
                print("Estimated rotation angle:", t)
        else:
            print("Cannot estimate rotation angle.")
    else:
        print("Not enough good matches.")


def extract_sift_orientations(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 计算关键点和描述符
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # 提取关键点的梯度方向
    orientations = [kp.angle for kp in keypoints]

    return orientations

def best_orientation(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 计算关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 初始化BFMatcher（Brute Force Matcher）
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用Lowe's比率测试（筛选好的匹配点）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 按匹配距离排序
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    if len(good_matches) > 0:
        best_match = good_matches[0]
        best_orientation1 = kp1[best_match.queryIdx].angle
        best_orientation2 = kp2[best_match.trainIdx].angle

        return best_orientation1,best_orientation2
    else:
        return None

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
    imgpath.append(img4path)
    imgpath.append(img3path)
    imgpath.append(img2path)
    imgpath.append(img1path)


    #compute_distances(img2path,img1path)
    #compute_distances_withSIFT(img5pathAngle,img6pathAngle)

    #best_orientation1,best_orientation2 = best_orientation(img5pathAngle,img6pathAngle)
    #print(best_orientation1,best_orientation2)

    #estimate_rotation(img1path,img2path)

    # distancex, distancey, gradient_direction1, gradient_direction2 = compute_distances(img1path,img2path)
    # print(distancex, distancey, gradient_direction1, gradient_direction2)

    #stitch_images_with_shift_multi(imgpath,outputpath2)
    stitch_images_with_shift_and_multi(imgpath,outputpath2)

    #stitch_images_with_shift(img3path, img2path, outputpath)
    print("complete")