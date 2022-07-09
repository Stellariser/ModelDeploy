import io

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
from model import model_factory
import cv2
import base64
from InversePerspective import PerspectiveTransform



app = flask.Flask(__name__)
model = None
use_gpu = False

palette = array([1,50])

def load_model():
    global model
    model = model_factory['bisenetplus'](2,aux_mode='pred')
    model.load_state_dict(torch.load('./bsp.pth', map_location='cpu'), strict=False)
    model.eval()


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


@app.route("/predict",methods = ["POST"])
def predict():
    data = {'success':False}
    img = request.form.get('img')
    if img:
        # 解码图像数据
        img = base64.b64decode(img.encode('ascii'))
        image_data = np.fromstring(img, np.uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        #cv2.imwrite('/home/wangdx/research/mir_robot/grasp_server/1.png', image_data)
        cv2.imwrite('./terminal/1.png', image_data)
        image = Image.open('./terminal/1.png')
        image = prepare_image(image, traget_size=(512, 1024))

        out = model(image).squeeze().detach().cpu().numpy()
        pred = palette[out]
        cv2.imwrite('./res/res.png', pred)
        data["success"] = True

        space = PerspectiveTransform(1024,512)

        data["Space"]=space

    return flask.jsonify(data)
    # data = {"success":False}
    # img = base64.b64decode(str(request.form['image']))
    # image_data = np.fromstring(img, np.uint8)
    # image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # cv2.imwrite('/terminal/1.png', image_data)

    #数据读取
    # image = flask.request.files["image"].read()
    # image = Image.open(io.BytesIO(image))
    #预处理

if __name__ == '__main__':
    print("loading model and start the server")
    load_model()
    app.run()