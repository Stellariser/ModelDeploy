import cv2
import numpy as np
from matplotlib import pyplot as plt

left = cv2.imread("./ContinusePics/2.jpg")
right = cv2.imread("./ContinusePics/1.jpg")

gray1 = cv2.cvtColor(left, cv2.THRESH_BINARY)
gray2 = cv2.cvtColor(right, cv2.THRESH_BINARY)

#创建SIFT对象
sift = cv2.SIFT_create()

#有两个必须的参数，第一个是求解特征点和特征向量的图像。第二个参数决定是否对特定区域求解，传入一个mask。
#返回值kps是对图像求解的特征点，是一个一维向量，其中每一个元素属于keypoint类型。dp是与kps对性的特征向量，也是一个列表，其中每一个元素是长度为128的向量.
kpsA, dpA = sift.detectAndCompute(gray1, None)
kpsB, dpB = sift.detectAndCompute(gray2, None)

#大力出奇迹
bf = cv2.BFMatcher()

matches = bf.knnMatch(dpA, dpB, k=2)
#去除不可靠的匹配
good_matches = []
for m in matches:
    if len(m) == 2 and m[0].distance < 0.4 * m[1].distance:
        good_matches.append((m[0].queryIdx, m[0].trainIdx))

kps1 = np.float32([kp.pt for kp in kpsA])
kps2 = np.float32([kp.pt for kp in kpsB])
kps1 = np.float32([kps1[a[0]] for a in good_matches])
kps2 = np.float32([kps2[a[1]] for a in good_matches])

M, status = cv2.findHomography(kps2, kps1, cv2.RANSAC, 4.0)
result = cv2.warpPerspective(right, M, (left.shape[1] + right.shape[1], right.shape[0]))

result[0:left.shape[0], 0:left.shape[1]] = left

# cv2.imshow("left", left)
# cv2.imshow("right", right)
# cv2.imshow("result", result)

plt.subplot(221), plt.imshow(left), plt.title('left')
plt.subplot(222), plt.imshow(right), plt.title('right')
plt.subplot(223), plt.imshow(result), plt.title('result')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()