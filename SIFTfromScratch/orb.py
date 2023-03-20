import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import os

# _______________________________#

img1 = cv2.imread('./transformedPic/1.png')
img2 = cv2.imread('./transformedPic/2.png')


def stitch_images(img1_path, img2_path, output_path, vertical_separation=0, horizontal_separation=0):
    # Open the input images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Calculate the size of the output image
    width = img1.width + img2.width + horizontal_separation
    height = max(img1.height, img2.height) + vertical_separation

    # Create a new image with the combined dimensions
    stitched_image = Image.new('RGB', (width, height))

    # Paste the first image onto the new image
    stitched_image.paste(img1, (0, 0))

    # Paste the second image with the specified horizontal and vertical separation
    stitched_image.paste(img2, (img1.width + horizontal_separation, vertical_separation))

    # Save the stitched image
    stitched_image.save(output_path)


def compute_distances(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    img = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Matching algorithm
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    distancey = float(abs(kp1[matches[0].queryIdx].pt[1] - kp2[matches[0].trainIdx].pt[1]))
    distancex = float(abs(kp1[matches[0].queryIdx].pt[0] - kp2[matches[0].trainIdx].pt[0]))

    return distancex, distancey


distancex, distancey = compute_distances('./transformedPic/1.png', './transformedPic/2.png')
print(distancex,distancey)

rows, cols = img1.shape[:2]
res = np.zeros([rows, cols, 3], np.uint8)
for a in range(0, rows):
    for b in range(0, cols):
        if not img1[a, b].any():
            res[a, b] = img2[a, b]
        elif not img2[a, b].any():
            res[a, b] = img1[a, b]
        else:
            res[a, b] = np.clip(img1[a, b] + img2[a, b], 0, 255)

cv2.imwrite("./resaaa.png", res)
# print(matches[i].queryIdx)
# knn = KNeighborsClassifier(n_neighbors=1)  # 使用K临近，若图1中的一个描述符的位置在图二中周围有1个临近点，则将这两个临近点匹配
# knn.fit(des1, [0] * len(des1))  # 生成长度为len(descriptors)全是0的数组作为标签，训练一个knn分类器[所有点都为0类]
# match = knn.kneighbors(des2, n_neighbors=1, return_distance=True)  # 匹配，找1个临近点即为一类，把index和distance存入match中


#img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:5], None, flags=2)
# cv2.imshow('Keypoints', img3)
#cv2.imwrite("./imgs/res.png", img3)

# plt.imshow(img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
