from stitching import AffineStitcher
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pathlib import Path
from stitching.image_handler import ImageHandler
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching import AffineStitcher
from stitching import Stitcher

def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def plot_images(imgs, figsize_in_inches=(5, 5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]


settings = {# The whole plan should be considered
            "crop": False,
            # The matches confidences aren't that good
            "confidence_threshold": 0.3}

weir_imgs = get_image_paths('weir')
budapest_imgs = get_image_paths('buda')
exposure_error_imgs = get_image_paths('exp')

img_handler = ImageHandler()
img_handler.set_img_names(weir_imgs)

medium_imgs = list(img_handler.resize_to_medium_resolution())
low_imgs = list(img_handler.resize_to_low_resolution(medium_imgs))
final_imgs = list(img_handler.resize_to_final_resolution())

original_size = img_handler.img_sizes[0]
medium_size = img_handler.get_image_size(medium_imgs[0])
low_size = img_handler.get_image_size(low_imgs[0])
print(AffineStitcher.AFFINE_DEFAULTS)

imgs = budapest_imgs

stitcher = Stitcher(compensator="no", blender_type="no",confidence_threshold = 0.3)
panorama1 = stitcher.stitch(imgs)

stitcher = Stitcher(compensator="no",confidence_threshold = 0.3)
panorama2 = stitcher.stitch(imgs)

stitcher = Stitcher(compensator="no", blend_strength=20,confidence_threshold = 0.3)
panorama3 = stitcher.stitch(imgs)

stitcher = Stitcher(blender_type="no",confidence_threshold = 0.3)
panorama4 = stitcher.stitch(imgs)

stitcher = Stitcher(blend_strength=20,confidence_threshold = 0.3)
panorama5 = stitcher.stitch(imgs)

fig, axs = plt.subplots(3, 2, figsize=(20,20))
axs[0, 0].imshow(cv.cvtColor(panorama1, cv.COLOR_BGR2RGB))
axs[0, 0].set_title('Along Seam Masks with Exposure Errors')
axs[0, 1].axis('off')
axs[1, 0].imshow(cv.cvtColor(panorama2, cv.COLOR_BGR2RGB))
axs[1, 0].set_title('Blended with the default blend strenght of 5')
axs[1, 1].imshow(cv.cvtColor(panorama3, cv.COLOR_BGR2RGB))
axs[1, 1].set_title('Blended with a bigger blend strenght of 20')
axs[2, 0].imshow(cv.cvtColor(panorama4, cv.COLOR_BGR2RGB))
axs[2, 0].set_title('Along Seam Masks with Exposure Error Compensation')
axs[2, 1].imshow(cv.cvtColor(panorama5, cv.COLOR_BGR2RGB))
axs[2, 1].set_title('Blended with Exposure Compensation and bigger blend strenght of 20')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()