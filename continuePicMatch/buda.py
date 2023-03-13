import cv2
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



weir_imgs = get_image_paths('weir')
budapest_imgs = get_image_paths('buda')
exposure_error_imgs = get_image_paths('exp')

# img_handler = ImageHandler()
# img_handler.set_img_names(weir_imgs)

# medium_imgs = list(img_handler.resize_to_medium_resolution())
# low_imgs = list(img_handler.resize_to_low_resolution(medium_imgs))
# final_imgs = list(img_handler.resize_to_final_resolution())
#
# original_size = img_handler.img_sizes[0]
# medium_size = img_handler.get_image_size(medium_imgs[0])
# low_size = img_handler.get_image_size(low_imgs[0])
# print(AffineStitcher.AFFINE_DEFAULTS)
print(AffineStitcher.AFFINE_DEFAULTS)
settings = {# The whole plan should be considered
            "crop": False ,
            # The matches confidences aren't that good
            "confidence_threshold": 0.41,
             }

stitcher = AffineStitcher(**settings)
panorama = stitcher.stitch(budapest_imgs)

plot_image(panorama, (20,20))