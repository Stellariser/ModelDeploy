import argparse
import io
from flask_cors import *
import numpy as np
from numpy import array
import flask as flask
from flask import request
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50
from torch.autograd import Variable

from configs import set_cfg_from_file
from model import model_factory
import cv2
import base64
from InversePerspective import PerspectiveTransform

app = flask.Flask(__name__)
CORS(app)
model = None
use_gpu = False
palette = array([1,60])
#palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
print(palette)
def load_model():
    global model
    model = model_factory['bisenetplus'](2,aux_mode='pred')
    model.load_state_dict(torch.load('./bsp.pth', map_location='cpu'), strict=False)
    model.eval()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--configs', dest='configs', type=str,
                        default='./configs/default.py',)
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

@app.route("/predict",methods = ["POST","GET"])
def predict():
    data = {'success':False}

    print(request.form.get('img'))
    img = request.form.get('img')

    useConfig= request.form.get('useConfig')


    if img:
        # 解码图像数据
        img = base64.b64decode(img.encode('ascii'))
        image_data = np.fromstring(img, np.uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        cv2.imwrite('./terminal/1.png', image_data)
        image = Image.open('./terminal/1.png')
        image = prepare_image(image, traget_size=(512, 1024))

        out = model(image).squeeze().detach().cpu().numpy()
        pred = palette[out]
        cv2.imwrite('./res/res.png', pred)
        data["success"] = True

        if useConfig == 'True':
            camera_angle = int(request.form.get('camera_angle'))
            inside_angle = int(request.form.get('inside_angle'))
            height = int(request.form.get('height'))
            originH = int(request.form.get('originH'))
            originW = int(request.form.get('originW'))
            space = PerspectiveTransform(camera_angle,inside_angle,height,originH,originW)
        else:
            args = parse_args()
            cfg = set_cfg_from_file(args.configs)
            camera_angle = cfg.camera_angle
            inside_angle = cfg.inside_angle
            height = cfg.height
            originH = cfg.originH
            originW = cfg.originW
            space = PerspectiveTransform(camera_angle,inside_angle,height,originH,originW)

        data["Space"]=space[0]
        data["Cany"]=space[1]
    return flask.jsonify(data)


if __name__ == '__main__':
    print("loading model and start the server")
    load_model()
    app.run(host='127.0.0.1', port=5000, threaded=True)