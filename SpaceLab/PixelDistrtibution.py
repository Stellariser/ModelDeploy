import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def statistics():
    src = cv.imread("../SpaceLab/res.png")

    h,w,ch = np.shape(src)
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)

    hest = np.zeros([256],dtype = np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray[row,col]
            hest[pv] +=1
    arr = np.unique(hest)
    arr[0] = arr[1]+arr[2]
    percentage = arr[2]/arr[0]
    return percentage

if __name__ == '__main__':
    statistics()