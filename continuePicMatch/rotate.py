import cv2

R = cv2.imread('D:/ICS/ModelDeploy/continuePicMatch/imgs/buda_4.png')
R = cv2.rotate(R,cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('D:/ICS/ModelDeploy/continuePicMatch/rotatebuda/buda_4.png',R)
