import cv2
import numpy as np
from matplotlib import pyplot as plt
import math



def PerspectiveTransform(W,H):
    camera_angle = 35  # alpha 相机对地法线角度
    inside_angle = 40  # bata  相机内视场纵向角度
    height = 40  # 飞行高度
    camera_with = camera_angle * 1.79  # 相机内视场横向角度
    tiangle = (180 - camera_with) / 2  # 上横向视场与地面夹角

    originH = 512
    originW = 1024

    alin = math.radians(camera_with)
    ti = math.radians(tiangle)
    al = math.radians(camera_angle)
    ba = math.radians(inside_angle)

    origin_length = ((height * math.cos(ba) * math.cos(ba / 2)) / (math.cos(al))) / math.sin(math.pi / 2 - al - ba)
    result_length = ((height * math.cos(ba) * math.cos(ba / 2)) / (math.cos(al)))

    squeeze_rate = result_length / origin_length
    bottom = math.tan(alin / 2) * height / math.cos(al) * 2
    head = bottom + origin_length / math.tan(ti)

    result_pic_reso = math.floor(origin_length / head * originW)

    pixavg = originW / head

    x1 = ((head - bottom) / 2) * pixavg
    x2 = ((head - bottom) / 2 + bottom) * pixavg

    img = cv2.imread('./res/res.png')

    pts1 = np.float32([[0, 0], [originW, 0], [originW, originH], [0, originH]])
    pts2 = np.float32([[0, 0], [originW, 0], [x2, result_pic_reso], [x1, result_pic_reso]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (originW, result_pic_reso))

    cv2.imwrite('./TransformedMask/trm.png', dst)


    h, w, ch = np.shape(dst)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    hest = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray[row, col]
            hest[pv] += 1
    arr = np.unique(hest)
    arr[0] = arr[1] + arr[2]
    percentage = arr[2] / arr[0]


    Space = (((head + bottom) * origin_length) / 2)*percentage


    return Space

def a():
    src = cv2.imread("../SpaceLab/res.png")

    h,w,ch = np.shape(src)
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    hest = np.zeros([256],dtype = np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray[row,col]
            hest[pv] +=1
    arr = np.unique(hest)
    arr[0] = arr[1]+arr[2]
    percentage = arr[2]/arr[0]
    return percentage