import requests
import argparse
import base64
import cv2

Pytorch_REST_API_URL = 'http://124.222.108.174:8080/upload'
#Pytorch_REST_API_URL = 'http://localhost:8080/upload'

imagePath = './Testimage/zhatuche.jpg'

files = {'file': open('./Testimage/zhatuche.jpg', 'rb')}

def predict_result(image_path):
    data = {
        'station': "1",
        'position' : "east",
        'AlertText' : "dump",
        'AlertType' : "dump",
        }
    r = requests.post(Pytorch_REST_API_URL,files=files, data=data)
    result = r.text

    print(result)


if __name__ == '__main__':

    predict_result(imagePath)
