import cv2 
import os
import numpy as np
from lib.detector import *

DIRNAME = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(DIRNAME,"data")
IMG_ANNOTATION_PATH = os.path.join(DIRNAME,"data_annotation.csv")

def read_images():
    images = {}
    for filename in os.listdir(IMG_PATH):

        img = cv2.imread(os.path.join(IMG_PATH,filename),cv2.IMREAD_GRAYSCALE)
        if(img is not None):
            images[filename] = img
        else:            
            print("Error with file", filename)
    return images

def main():
    #Read images
    images = read_images()

    #Image marbre
    #img = images["IMG_1651.JPG"]
   
    #Image test
    img = images["ImgPiece10.jpeg"]
    img = images["2.jpg"]
    
    blur = iterative_gaussian_blur(img,5)
    
    #Otsu 
    _, otsu_thresholded_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    
    #OPENING
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform erosion
    eroded_image = cv2.erode(otsu_thresholded_img, kernel, iterations=30)

    # Perform dilation
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=30)
    


    
    cv2.imshow("open img", dilated_image)
    

    #scale
    circles = cv2.HoughCircles(dilated_image, cv2.HOUGH_GRADIENT, dp=1, minDist=140,
                            param1=10, param2=20, minRadius=20, maxRadius=800)

    if circles is not None:
        # Calculate average size of detected circles
        average_radius = np.mean(circles[0, :, 2])

        # Choose a reference diameter (you can adjust this based on your requirements)
        reference_diameter = 100

        # Calculate scale factor
        scale_factor = reference_diameter / (2 * average_radius)

        # Resize the image using the scale factor
        resized_image = cv2.resize(dilated_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Display the resized image
        cv2.imshow("Resized Image", resized_image)
        cv2.imshow("Original Image", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles found in the image.")
    
    
    #Circle 
    circles = cv2.HoughCircles(resized_image, cv2.HOUGH_GRADIENT, 1, 40,
                        param1=120, param2=25, minRadius=20, maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        cimg = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('detected circles',cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


    
if __name__ == "__main__":
    main()
