import cv2
import numpy as np


def findCoin(image):
    """
    Detect circles in the input image using Hough Circle Transform.

    Args:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        tuple: A tuple containing the color image with detected circles
               and the detected circles themselves.
    """

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100,
                        param1=60, param2=20, minRadius=9, maxRadius=150)
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    return cimg,circles

    
def iterative_gaussian_blur(image, num_iterations=3, start_sigma=0, step=1):
    """
    Apply iterative Gaussian blur to the input image.

    Args:
        image (numpy.ndarray): Input grayscale image.
        num_iterations (int): Number of iterations.
        start_sigma (int): Initial sigma value.
        step (int): Step size for sigma.

    Returns:
        numpy.ndarray: Blurred image.
    """
    blurred_image = image.copy()
    sigma = start_sigma
    for _ in range(num_iterations):
        blurred_image = cv2.GaussianBlur(blurred_image, (5, 5), sigma)
        sigma += step
    return blurred_image

def pretreat(img):
    """
    Preprocess the input image.

    Args:
        img (numpy.ndarray): Input color image.

    Returns:
        numpy.ndarray: Preprocessed grayscale image.
    """

    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.GaussianBlur(res, (21, 21),2)
    return res

def find_background_color(image,border_size=5):
    """
    Estimate the background color of the input image.

    Args:
        image (numpy.ndarray): Input grayscale image.
        border_size (int): Size of the border to consider.

    Returns:
        float: Estimated background color.
    """   

    h, w = image.shape
    b = border_size
    top_border, bottom_border, left_border, right_border = image[:b,:], image[h-b:h,:], image[:,:b], image[:,w-b:w]
    m = np.mean((np.mean(top_border), np.mean(bottom_border), np.mean(left_border), np.mean(right_border))) / 4

    return m

def apply_threshold(img):
    """
    Apply Otsu's thresholding to the input image.

    Args:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Thresholded image.
    """

    res = img.copy()
    ret, thresh = cv2.threshold(res,0,255,cv2.THRESH_OTSU)
    if find_background_color(thresh) >= 127 :
        ret, inv_thresh = cv2.threshold(res,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        res = inv_thresh
    else: 
        res = thresh

    return res

def find_background(img):
    """
    Estimate the background of the input image.

    Args:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Estimated background image.
    """

    res = img.copy()
    res = cv2.medianBlur(res, 3)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.dilate(res,kernel,iterations=3)
    return bg

def find_centroids(img):
    """
    Find centroids in the input image.

    Args:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Image with centroids.
    """
    res = img.copy()
    dist_transform = cv2.distanceTransform(res,cv2.DIST_L2,5)
    ret, fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    fg = np.uint8(fg)
    return fg


def segment_with_watershed(img):
    """
    Segment objects in the input image using watershed algorithm.

    Args:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Segmented image.
    """
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
