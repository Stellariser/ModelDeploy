import numpy as np
import cv2
currentframe = 0
fps = 1
FrameFrequency = fps*1
interval =50

# 第一步：视频的读入
cap = cv2.VideoCapture("./video/v1_Trim.mp4")
video = cv2.VideoCapture("./video/v1.mp4")
#ret, frame = video.read()
# 第二步：构建角点检测所需参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=100)

# lucas kanade参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2)

# 随机颜色条
color = np.random.randint(0, 255, (100, 3))

# 第三步：拿到第一帧图像并灰度化作为前一帧图片
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# 第四步:返回所有检测特征点，需要输入图片，角点的最大数量，品质因子，minDistance=7如果这个角点里有比这个强的就不要这个弱的
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 第五步:创建一个mask, 用于进行横线的绘制
mask = np.zeros_like(old_frame)

while(True):
    currentframe += 1
    ret, frame = cap.read()
    if currentframe % FrameFrequency == 0:
    # 第六步：读取图片灰度化作为后一张图片的输入

        if currentframe % interval == 0:
            terminalframe = frame
            old_gray = cv2.cvtColor(terminalframe, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(terminalframe)
            print(mask,"new")

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 第七步：进行光流检测需要输入前一帧和当前图像及前一帧检测到的角点
        pl, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 第八步：读取运动了的角点st == 1表示检测到的运动物体，即v和u表示为0
        good_new = pl[st==1]
        good_old = p0[st==1]

        # # 第九步：绘制轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        # 第十步：将两个图片进行结合，并进行图片展示

        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(150) & 0xff
        if k == 27:
            break

        # 第十一步：更新前一帧图片和角点的位置
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cv2.destroyAllWindows()
cap.release()