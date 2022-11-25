import io
import sys

import cv2
import flask as flask
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from numpy import array
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50
from res import *
from torchsummary import summary
from model import bisenetplus
from model import model_factory
from torch.autograd import Variable


model = None
model2 = None
use_gpu = False

#palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette = array([1,60])


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

def load_model():
    global model

    model = model_factory['bisenetplus'](2,aux_mode='pred')
    model.load_state_dict(torch.load('./bsp.pth', map_location='cpu'), strict=False)
    model.eval()

    #summary(model, (3, 512, 1024))


    # if use_gpu:
    #     model.cuda()
def predict():

    image = getimage()
    out = model(image).squeeze().detach().cpu().numpy()
    pred = palette[out]
    cv2.imwrite('./res.jpg', pred)

def getimage():
    image = Image.open('./Testimage/18.png')
    image = prepare_image(image, traget_size=(512, 1024))
    return image

if __name__ == '__main__':
    load_model()
    predict()