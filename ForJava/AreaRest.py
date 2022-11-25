import argparse
import os
import sys

from numpy import array
import torch
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable
import cv2

from InversePerspective import PerspectiveTransform
from configs import set_cfg_from_file

model = None
use_gpu = False
palette = array([1,60])

from model import model_factory

file_path = os.path.abspath(__file__)
cur_path = os.path.dirname(file_path)
project_path = os.path.dirname(cur_path)

sys.path.append(project_path)

sys.path.append("D:\ICS\ModelDeploy\ForJava")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--filepath', type=str, default = None)
args = parser.parse_args()



def load_model():
    global model
    model = model_factory['bisenetplus'](2,aux_mode='pred')
    model.load_state_dict(torch.load('./Weighting/bsp.pth', map_location='cpu'), strict=False)
    model.eval()

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--configs', dest='configs', type=str,
                        default='./configs/default.py',)
    parse.add_argument('--filepath', dest='filepath', type=str,
                       default='./terminal/12.png', )
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


if __name__ == '__main__':

    args = parse_args()
    filepath = args.filepath
    load_model()
    data = {'success': False}
    useConfig = False
    imagePath = filepath

    image = Image.open(imagePath)
    image = prepare_image(image, traget_size=(512, 1024))

    out = model(image).squeeze().detach().cpu().numpy()
    pred = palette[out]
    cv2.imwrite('./res/res.png', pred)

    if useConfig == 'True':
        args = parse_args()
        cfg = set_cfg_from_file(args.configs)
        camera_angle = cfg.camera_angle
        inside_angle = cfg.inside_angle
        height = cfg.height
        originH = cfg.originH
        originW = cfg.originW
        # camera_angle = int(request.form.get('camera_angle'))
        # inside_angle = int(request.form.get('inside_angle'))
        # height = int(request.form.get('height'))
        # originH = int(request.form.get('originH'))
        # originW = int(request.form.get('originW'))
        space = PerspectiveTransform(camera_angle, inside_angle, height, originH, originW)
    else:
        args = parse_args()
        cfg = set_cfg_from_file(args.configs)
        camera_angle = cfg.camera_angle
        inside_angle = cfg.inside_angle
        height = cfg.height
        originH = cfg.originH
        originW = cfg.originW
        space = PerspectiveTransform(camera_angle, inside_angle, height, originH, originW)

    data["Space"] = space[0]
    data["Cany"] = space[1]
    print("Space",space[0],"@","Cany",space[1])

