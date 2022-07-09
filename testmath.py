import numpy as np
from matplotlib import pyplot as plt
import math

camera_angle = 30   #alpha
camera_with = camera_angle*1.79
inside_angle = 40   #bata
height = 40
tiangle = 60

al = math.radians(camera_angle)
alin = math.radians(camera_with)
ba = math.radians(inside_angle)
ti = math.radians(tiangle)

origin_length = ((height*math.cos(ba)*math.cos(ba/2))/(math.cos(al)))/math.sin(math.pi/2-al-ba)
result_length = ((height*math.cos(ba)*math.cos(ba/2))/(math.cos(al)))

bottom = math.tan(alin/2)*70/math.cos(al)*2
head = bottom + 2*origin_length/math.tan(ti)
result_pic_reso = math.floor(origin_length/head*1920)

pixaverage = 1920/151
x1 = ((head-bottom)/2)*pixaverage
x2 = ((head-bottom)/2)*pixaverage+bottom
Space=((head+bottom)*origin_length)/2

print('result_length',result_length)
print('origin_length',origin_length)
print('bottom',bottom)
print('head',head)
print("result_pic_reso", result_pic_reso)
print("pixaverage", pixaverage)
print("x1",x1)
print("x2",x2)
print("Space",Space)