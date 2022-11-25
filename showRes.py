from matplotlib import pyplot as plt
import cv2

fig, ax = plt.subplots()
A = cv2.imread('./Testimage/2.png')  # 左半部分
B = cv2.imread('./res/res.png')  # 左半部分
C = cv2.imread('./TransformedMask/trm.png')  # 左半部分
D = cv2.imread('./Transformedpic/1.png')  # 左半部分

plt.subplot(221), plt.imshow(A), plt.title('Original image')
plt.subplot(222), plt.imshow(B), plt.title('Results')
plt.subplot(223), plt.imshow(D), plt.title('Original image trans')
plt.subplot(224), plt.imshow(C), plt.title('result trans')

plt.show()

fig.savefig('./res.svg',dpi=600,format='svg')