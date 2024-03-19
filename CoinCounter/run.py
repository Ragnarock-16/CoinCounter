import cv2
import os
import numpy as np
import pandas as pd

DIRNAME = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(DIRNAME, "data")
IMG_ANNOTATION_PATH = os.path.join(DIRNAME, "data_annotation.csv")


def read_images():
    images = {}
    for filename in os.listdir(IMG_PATH):
        img = cv2.imread(os.path.join(IMG_PATH, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename] = img
        else:
            print("Error with file", filename)
    return images


def main():
    # Read images
    images = read_images()

    # Image marbre
    img = images["IMG_1651.JPG"]

    # Image test
    # img = images["ImgPiece10.jpeg"]

    # img = cv2.GaussianBlur(img,(23,23),0)
    # _,thresh = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY_INV)

    # laplacian = cv2.Laplacian(thresh,cv2.CV_64F)
    # dil = cv2.dilate(laplacian,(1,1),3)
    # cv2.imshow("Test",img)
    # cv2.imshow("Thresh",thresh)
    # cv2.imshow("Laplacian",laplacian)
    # cv2.imshow("dil",dil)


# cv2.imwrite("test.jpg",thresh)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
