import cv2
import numpy as np
from matplotlib import pyplot as plt


def pretreat(img):
    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    res = cv2.resize(res, (512, 512), interpolation=cv2.INTER_AREA)
    res = cv2.GaussianBlur(res, (5, 5), 2)
    return res


def find_background_color(image, border_size=5):
    h, w = image.shape
    b = border_size
    top_border, bottom_border, left_border, right_border = (
        image[:b, :],
        image[h - b : h, :],
        image[:, :b],
        image[:, w - b : w],
    )
    m = (
        np.mean(
            (
                np.mean(top_border),
                np.mean(bottom_border),
                np.mean(left_border),
                np.mean(right_border),
            )
        )
        / 4
    )

    return m


def apply_threshold(img):
    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(res, 0, 255, cv2.THRESH_OTSU)
    num_white_pixels = np.sum(thresh == 255)
    num_black_pixels = np.sum(thresh == 0)
    if num_white_pixels > num_black_pixels:
        ret, inv_thresh = cv2.threshold(
            res, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        res = inv_thresh
    else:
        res = thresh
    res = cv2.medianBlur(res, 3)

    return res


def get_background(img):
    res = img.copy()
    res = cv2.medianBlur(res, 3)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.dilate(res, kernel, iterations=3)
    return bg


def get_centroids(img):
    res = img.copy()
    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    ret, fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    fg = np.uint8(fg)
    return fg


def segment_with_watershed(img):
    res = img.copy()
    res = apply_threshold(res)
    bg = get_background(res)
    fg = get_centroids(res)
    unknown = cv2.subtract(bg, fg)
    ret, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    num_regions = np.max(markers) - 1
    return markers, num_regions


def segment_with_hough_circles(img):

    pass


def count_coins():
    pass


def evaluate():
    pass
