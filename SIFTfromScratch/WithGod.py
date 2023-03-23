import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import os

# _______________________________#

img1 = cv2.imread('./transformedPic/1.png')
img2 = cv2.imread('./transformedPic/2.png')
img3 = cv2.imread('./transformedPic/3.png')
img4 = cv2.imread('./transformedPic/4.png')

img1path = './transformedPic/1.png'
img2path = './transformedPic/2.png'
img3path = './transformedPic/3.png'
img4path = './transformedPic/4.png'
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



    distancey = float(kp1[matches[0].queryIdx].pt[1] - kp2[matches[0].trainIdx].pt[1])
    distancex = float(kp1[matches[0].queryIdx].pt[0] - kp2[matches[0].trainIdx].pt[0])

    # print(distancex,"看这个",kp1[matches[0].queryIdx].pt[0],kp2[matches[0].trainIdx].pt[0])


    return distancex, distancey

def stitch_images_with_shift(img1_path, img2_path, output_path):


    x_shift, y_shift = compute_distances(img2_path, img1_path)
    print(x_shift, y_shift)

    x_shift = abs(int(x_shift))
    y_shift = abs(int(y_shift))

    # Open the input images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Calculate the size of the output image
    width = img1.width + 2*abs(x_shift)
    height = max(img1.height, img2.height + abs(y_shift))

    # Create a new image with the combined dimensions
    stitched_image = Image.new('RGB', (width, height))

    # Paste the first image onto the new image
    stitched_image.paste(img1, (x_shift, 0))

    # Paste the second image with the specified horizontal and vertical shifts
    stitched_image.paste(img2, (x_shift-x_shift, y_shift))

    # Save the stitched image
    stitched_image.save(output_path)

def laplacian_blending(img1, img2, x_shift, y_shift):
    # Generate Gaussian pyramids for both images
    gp1 = [img1]
    gp2 = [img2]
    for i in range(6):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        gp1.append(img1)
        gp2.append(img2)

    # Generate Laplacian pyramids for both images
    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]
    for i in range(6, 0, -1):
        img1 = cv2.pyrUp(gp1[i])
        img2 = cv2.pyrUp(gp2[i])

        img1_resized = cv2.resize(img1, (gp1[i - 1].shape[1], gp1[i - 1].shape[0]))
        img2_resized = cv2.resize(img2, (gp2[i - 1].shape[1], gp2[i - 1].shape[0]))

        lap1 = cv2.subtract(gp1[i - 1], img1_resized)
        lap2 = cv2.subtract(gp2[i - 1], img2_resized)
        lp1.append(lap1)
        lp2.append(lap2)

    # Combine the Laplacian pyramids using the weighted approach
    blended_lap = []
    for l1, l2 in zip(lp1, lp2):
        rows, cols, _ = l1.shape
        blend_mask = np.zeros((rows, cols), dtype=np.float32)
        blend_mask[:, :cols // 2 + x_shift] = 1.0

        # Create a 3-channel mask for blending
        blend_mask_3ch = np.repeat(blend_mask[..., None], 3, axis=2)

        # Blend the Laplacian levels using the mask
        blend = blend_mask_3ch * l1 + (1 - blend_mask_3ch) * l2
        blended_lap.append(blend)

    # Reconstruct the blended image from the combined pyramid
        # Reconstruct the blended image from the combined pyramid
        blended_image = blended_lap[0]
        for i in range(1, len(blended_lap)):
            blended_image = cv2.pyrUp(blended_image)

            # Ensure the same size before adding
            blended_lap_resized = cv2.resize(blended_lap[i], (blended_image.shape[1], blended_image.shape[0]))

            blended_image = cv2.add(blended_image, blended_lap_resized)

    return blended_image

def stitch_images_with_blending(img1_path, img2_path, output_path, x_shift, y_shift):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Calculate the size of the output image
    width = img1.shape[1] + x_shift
    height = max(img1.shape[0], img2.shape[0] + y_shift)

    # Create a new image with the combined dimensions
    stitched_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Copy the first image onto the new image
    stitched_image[:img1.shape[0], :img1.shape[1]] = img1

    # Apply the specified horizontal and vertical shifts to the second image
    shifted_image = np.zeros((height, width, 3), dtype=np.uint8)
    shifted_image[y_shift : y_shift + img2.shape[0], x_shift : x_shift + img2.shape[1]] = img2

    # Blend the images using Laplacian pyramid blending
    blended_image = laplacian_blending(stitched_image, shifted_image, x_shift, y_shift)

    # Save the stitched and blended image
    cv2.imwrite(output_path, blended_image)





if __name__ == '__main__':

    outputpath = './resaaa.png'
    outputpath2 = './resaaab.png'
    stitch_images_with_shift(img3path,img2path,outputpath)
    #stitch_images_with_blending(img2path,img1path,outputpath2,int(distancex),int(distancey))

    print('complete')