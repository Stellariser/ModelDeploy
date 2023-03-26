import math

import cv2
import numpy as np
from PIL import Image


def rotate_rgba_image(img, rotation_angle, output_path):
    # 读取图像
    #src_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 获取旋转矩阵
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 计算旋转后的图像尺寸
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(h * abs_sin + w * abs_cos)
    new_height = int(h * abs_cos + w * abs_sin)
# 更新旋转矩阵中心
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

# 旋转图像
    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

# 保存旋转后的图像
    cv2.imwrite(output_path, rotated_img)

def rotate_rgba_image_fixed_center(image_path, rotation_angle, output_path):
    # 读取图像
    src_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 获取旋转矩阵
    (h, w) = src_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 保持与原始图像相同的尺寸
    new_width = w
    new_height = h

    # 旋转图像
    rotated_img = cv2.warpAffine(src_img, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # 保存旋import cv2
import numpy as np

def rotate_rgba_image_fixed_center(img, rotation_angle, output_path):
    # 读取图像
    #src_img = Image.open(image_path).convert("RGBA")

    # 获取旋转矩阵
    (w, h) = img.size
    center = (w // 2, h // 2)

    # 计算新的图像尺寸
    angle_rad = math.radians(rotation_angle)
    new_width = int(abs(h * math.sin(angle_rad)) + abs(w * math.cos(angle_rad)))
    new_height = int(abs(h * math.cos(angle_rad)) + abs(w * math.sin(angle_rad)))

    # 创建一个新的透明图像，尺寸为新的宽度和高度
    new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

    # 将源图像粘贴到新图像的中心
    new_img.paste(img, (int((new_width - w) / 2), int((new_height - h) / 2)))

    # 旋转图像
    rotated_img = new_img.rotate(-rotation_angle, resample=Image.BICUBIC, expand=True)

    # 保存旋转后的图像
    rotated_img.save(output_path)




if __name__ == '__main__':
    src_img = Image.open('./bmp/1.png').convert("RGBA")
    rotate_rgba_image_fixed_center(src_img,30,'./rotatetestimg.png')