import argparse
from numpy import array
import torch
import cv2
import os
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable

from InversePerspective import PerspectiveTransform
from model import model_factory

model = None
use_gpu = False
palette = array([1, 60])
video = cv2.VideoCapture("./video/v1.mp4")
currentframe = 0
fps = 1
FrameFrequency = fps*25

url = 'rtsp://admin:Wurenchuan619@192.168.137.109:1554/h264/ch33/main/av_stream'
outputDir = './frames/'

def load_model():
    global model
    model = model_factory['bisenetplus'](2,aux_mode='pred')
    model.load_state_dict(torch.load('../bsp.pth', map_location='cpu'), strict=False)

    model.eval()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--configs', dest='configs', type=str,
                        default='../configs/default.py',)
    return parse.parse_args()

def prepare_image(image,traget_size):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    image = T.Resize(traget_size)(image)
    image = T.ToTensor()(image)

    image = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(image)
    #add Batchsize
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return Variable(image,volatile=True)

def showPic():
    cv2.namedWindow('Face', 0)
    cv2.resizeWindow('Face', 1080, 1920)
    cv2.moveWindow('Face', 100, 50)

cap = cv2.VideoCapture(url)

load_model()
while (True):
    currentframe += 1
    # reading from frame
    #ret, frame = video.read()
    ret, frame = cap.read()
    if not ret:
        print("Complete")
        break
    if currentframe%FrameFrequency==0:

        cv2.imwrite(outputDir + str(currentframe)+'.jpg', frame)

        frame = Image.open(outputDir + str(currentframe)+'.jpg')
        frame = prepare_image(frame, traget_size=(512, 1024))
        out = model(frame).squeeze().detach().cpu().numpy()
        pred = palette[out]
        cv2.imwrite('./res/res.png', pred)
        camera_angle = 29  # alpha 相机对地法线角度
        inside_angle = 29  # bata  相机内视场纵向角度
        height = 70  # 飞行高度
        originH = 512  # 分辨率
        originW = 1024

        space = PerspectiveTransform(camera_angle, inside_angle, height, originH, originW)
        cv2.imwrite(outputDir + str(currentframe)+'('+str(space)+')'+'m.jpg', pred)

# 一旦完成释放所有的空间和窗口
video.release()
cv2.destroyAllWindows()
