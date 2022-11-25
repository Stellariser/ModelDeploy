"""实现对图像的缝合（注意点：需要保证左右两张图像的size一样）"""

import cv2
import numpy as np
from PIL import Image
# 设置一个至少10个匹配的条件（有MinMatchNum指定）来找目标
from matplotlib import pyplot as plt

MinMatchNum = 20
fig, ax = plt.subplots()
# 读取照片
L = cv2.imread('./ContinusePics/9.jpg')  # 左半部分
L = cv2.rotate(L,cv2.ROTATE_90_COUNTERCLOCKWISE)


R = cv2.imread('./ContinusePics/8.jpg')  # 右半部分
R = cv2.rotate(R,cv2.ROTATE_90_COUNTERCLOCKWISE)
# 高斯滤波
L = cv2.GaussianBlur(L, (3, 3), 0)
R = cv2.GaussianBlur(R, (3, 3), 0)

# 创建sift检测器
sift = cv2.SIFT_create()

# 计算所有特征点的特征值kp和特征向量des并获取
left_kp, left_des = sift.detectAndCompute(R, None)
right_kp, right_des = sift.detectAndCompute(L, None)

# BFMatcher解决匹配，但是不好的特征值匹配较多
bf = cv2.BFMatcher()
matches = bf.knnMatch(left_des, right_des, k=2)
print(matches)

# 进行特征点匹配筛选
BetterChoose = []
for i, j in matches:
    # 认为第一近的点小于第二近的点一倍以上是好的匹配BetterChoose
    if i.distance < 0.5 * j.distance:
        BetterChoose.append(i)

# 使用Ransac优化匹配结果
BetterChooseAdd = np.expand_dims(BetterChoose, 1)
match = cv2.drawMatchesKnn(L, left_kp, R, right_kp, BetterChooseAdd[:30], None, flags=2)

# 判断是否当前模型已经符合超过MinMatchNum个点
if len(BetterChoose) > MinMatchNum:
    # 获取关键点的坐标
    src_pts = np.float32([left_kp[m.queryIdx].pt for m in BetterChoose]).reshape(-1, 1, 2)
    dst_pts = np.float32([right_kp[m.trainIdx].pt for m in BetterChoose]).reshape(-1, 1, 2)
    # 在这里调用RANSAC方法得到解H
    H, module = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    wrap = cv2.warpPerspective(R, H, (R.shape[1] + R.shape[1], R.shape[0] + R.shape[0]))
    wrap[0:R.shape[0], 0:R.shape[1]] = L
    # 得到新的位置
    rows, cols = np.where(wrap[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    # 去除黑色无用部分
    LeftAndRight = wrap[min_row:max_row, min_col:max_col, :]
    for i in range(0,3):
        LeftAndRight = np.rot90(LeftAndRight)
# 结果显示
scale = 0.5
# cv2.imshow('Left', cv2.resize(L, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))  # 左半部分
# cv2.imshow('Right', cv2.resize(R, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))  # 右半部分
# cv2.imshow('Match', cv2.resize(match, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))  # 匹配结果
# cv2.imshow('LeftAndRight', cv2.resize(LeftAndRight, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))  # 拼接结果

R = cv2.rotate(R,cv2.ROTATE_90_COUNTERCLOCKWISE)
R = cv2.rotate(R,cv2.ROTATE_90_COUNTERCLOCKWISE)
R = cv2.rotate(R,cv2.ROTATE_90_COUNTERCLOCKWISE)

L = cv2.rotate(L,cv2.ROTATE_90_COUNTERCLOCKWISE)
L = cv2.rotate(L,cv2.ROTATE_90_COUNTERCLOCKWISE)
L = cv2.rotate(L,cv2.ROTATE_90_COUNTERCLOCKWISE)

plt.subplot(221), plt.imshow(L), plt.title('1')
plt.subplot(222), plt.imshow(R), plt.title('2')
plt.subplot(223), plt.imshow(LeftAndRight), plt.title('1+2')
plt.show()
fig.savefig('./zipres.svg',dpi=600,format='svg')
cv2.waitKey(0)
cv2.destroyAllWindows()
