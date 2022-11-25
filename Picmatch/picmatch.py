
# 挑出两幅图像匹配的图像
import cv2 as cv
import numpy as np


def cut_resize(img1,img2):
    a1,b1 = img1.shape ; a2,b2 = img2.shape
    if a1>a2:
        a=a1
    else:
        a=a2
    if b1>b2:
        b=b1
    else:
        b=b2
    img_1c = cv.resize(img1,(b,a),interpolation = cv.INTER_CUBIC)
    img_2c = cv.resize(img2,(b,a),interpolation = cv.INTER_CUBIC)
    cv.imwrite('img_1.png',img_1c)
    cv.imwrite('img_2.png',img_2c)
    return img_1c,img_2c



# 读取两幅图像
img1 = cv.imread('./ContinusePics/1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./ContinusePics/2.jpg', cv.IMREAD_GRAYSCALE)

# 展示重叠部分
# img1, img2 = cut_resize(img_2, img_1)  # 插值到相同大小
overlap = cv.addWeighted(img1, 1, img2, 1, 0)
cv.imwrite('./overlap/overlap.png', overlap)

# 获得当前图片得权重矩阵，有值的地方为1，nan为0
w1 = np.copy(img1)
w1[np.isnan(img1)] = 0
w1[np.nonzero(img1)] = 1

w2 = np.copy(img2)
w2[np.isnan(img2)] = 0
w2[np.nonzero(img2)] = 1

# 将两幅图像的权重矩阵相乘
w = w1 * w2

# 取出两幅图公共部分
Img1 = img1 * w;
Img2 = img2 * w
cv.imwrite('./res/Img1.png', Img1)
cv.imwrite('./res/Img2.png', Img2)