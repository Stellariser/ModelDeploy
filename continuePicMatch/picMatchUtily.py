# 对那些旋转较大角度的图片去重

import cv2
import numpy as np


#获取图片关键点和特征向量
def detectAndDescribe(image):
    # 将彩色图片转成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT生成器
    destriptor = cv2.SIFT_create()
    kps, features = destriptor.detectAndCompute(gray, None)

    # 结果转成numpy数组
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)

#特征匹配
def matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
    # 建立暴力匹配器
    matcher = cv2.BFMatcher()

    # KNN检测来自两张图片的SIFT特征匹配对
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    #元组类型，924对
    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio时，保留此配对
        # (<DMatch 000001B1D6B605F0>, <DMatch 000001B1D6B60950>) 表示对于featuresA中每个观测点，得到的最近的来自B中的两个关键点向量
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append([m[0].trainIdx, m[0].queryIdx])
            # 这里怎么感觉只用了m[0]也就是最近的那个向量啊，应该没用到次向量
            # 这个m[0].trainIdx表示的时该向量在B中的索引位置， m[0].queryIdx表示的时A中的当前关键点的向量索引

    # 当筛选后的匹配对大于4时，可以拿来计算视角变换矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        #主要逻辑是从图片B中给图片A中的关键点拿最近的K个匹配向量，然后基于规则筛选，
        # 保存好匹配好的关键点的两个索引值，通过索引值取到匹配点的坐标值，
        # 有了多于4对的坐标值，就能得到透视变换矩阵。 这里返回的主要就是那个变换矩阵。
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # 计算视角变换矩阵  这里就是采样，然后解方程得到变换矩阵的过程
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return (matches, H, status)

    # 匹配结果小于4时，返回None
    return None

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # 返回可视化结果
    return vis



#读取两张有很大旋转的相似图片
image1 = cv2.imread('./picHD/1.png')
image2 = cv2.imread('./picHD/2.png')

# 检测A， B图片的SIFT特征关键点，得到关键点的表示向量
(kps_img1, features_img1) = detectAndDescribe(image1)
# kpsA (关键点个数， 坐标)  features(关键点个数，向量)
#kps_img1 (924, 2) features_img1  (924, 128)
(kps_img2, features_img2) = detectAndDescribe(image2)

# 匹配两张图片的所有特征点，返回匹配结果 注意，这里是变换right这张图像，所以应该是从left找与right中匹配的点，然后去计算right的变换矩阵
M = matchKeyPoints(kps_img1, kps_img2, features_img1, features_img2)
if M:
    # 提取匹配结果
    (matches, H, status) = M
    print('888888')
    vis = drawMatches(image1, image2, kps_img1, kps_img2, matches, status)
    cv2.imshow("vis",vis)
    cv2.waitKey()
    cv2.destroyAllWindows()
