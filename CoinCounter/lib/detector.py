import cv2
import numpy as np

def iterative_gaussian_blur(image, num_iterations=3, start_sigma=0, step=1):
    blurred_image = image.copy()
    sigma = start_sigma
    for _ in range(num_iterations):
        blurred_image = cv2.GaussianBlur(blurred_image, (5, 5), sigma)
        sigma += step
    return blurred_image

def pretreat(img):
    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.GaussianBlur(res, (21, 21),2)
    return res

def find_background_color(image,border_size=5):
    h, w = image.shape
    b = border_size
    top_border, bottom_border, left_border, right_border = image[:b,:], image[h-b:h,:], image[:,:b], image[:,w-b:w]
    m = np.mean((np.mean(top_border), np.mean(bottom_border), np.mean(left_border), np.mean(right_border))) / 4

    return m

def apply_threshold(img):
    res = img.copy()
    ret, thresh = cv2.threshold(res,0,255,cv2.THRESH_OTSU)
    if find_background_color(thresh) >= 127 :
        ret, inv_thresh = cv2.threshold(res,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        res = inv_thresh
    else: 
        res = thresh

    return res

def find_background(img):
    res = img.copy()
    res = cv2.medianBlur(res, 3)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.dilate(res,kernel,iterations=3)
    return bg

def find_centroids(img):
    res = img.copy()
    dist_transform = cv2.distanceTransform(res,cv2.DIST_L2,5)
    ret, fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    fg = np.uint8(fg)
    return fg


def segment_with_watershed(img):
    res = img.copy()
    fg = find_centroids(img)
    bg = find_background(img)
    unknown = cv2.subtract(bg,fg)
    ret, markers = cv2.connectedComponents(fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    res[markers == -1] = [255,0,0]
    return res

def segment_with_hough_circles(img):

    pass

def count_coins():
    pass


def evaluate():
    pass