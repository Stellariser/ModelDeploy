import requests
import argparse
import base64
import cv2

from ClientConfigs import set_cfg_from_file_client

Pytorch_REST_API_URL = 'http://127.0.0.1:5000/predict'

imagePath = './Testimage/2.png'

useConfig = True

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--configs', dest='configs', type=str,
                        default='./ClientConfigs/default.py',)
    return parse.parse_args()

def predict_result(image_path):
    img = cv2.imread(image_path)
    success, encoded_image = cv2.imencode(".jpg", img)
    img_bytes = encoded_image.tobytes()
    img_bytes = base64.b64encode(img_bytes)
    img_bytes = img_bytes.decode('ascii')
    if useConfig:
        args = parse_args()
        cfg = set_cfg_from_file_client(args.configs)
        data = {'img': img_bytes,
                'useConfig': True,
        'camera_angle' : cfg.camera_angle,
        'inside_angle' : cfg.inside_angle,
        'height' : cfg.height,
        'originH' : cfg.originH,
        'originW' : cfg.originW
        }
    else:
        data = {'img': img_bytes,
                'useConfig': False,
        }

    r = requests.post(Pytorch_REST_API_URL, data=data)
    result = r.text

    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file',type = str,help='test image files' )
    args = parser.parse_args()
    predict_result(imagePath)
