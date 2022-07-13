import requests
import argparse
import base64
import cv2
Pytorch_REST_API_URL = 'http://127.0.0.1:5000/predict'

imagePath = './Testimage/3.png'

def predict_result(image_path):
    img = cv2.imread(image_path)
    success, encoded_image = cv2.imencode(".jpg", img)
    img_bytes = encoded_image.tobytes()
    img_bytes = base64.b64encode(img_bytes)
    img_bytes = img_bytes.decode('ascii')
    data = {'img': img_bytes}
    r = requests.post(Pytorch_REST_API_URL, data=data)
    result = r.text

    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file',type = str,help='test image files' )
    args = parser.parse_args()
    predict_result(imagePath)
