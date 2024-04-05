import cv2 
import os
import numpy as np
from lib.detector import *
from lib.classifier import Classifier

DIRNAME = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(DIRNAME,"data")
IMG_ANNOTATION_PATH = os.path.join(DIRNAME,"data_annotation.csv")
myClassifier = Classifier()

def read_images():
    images = {}
    for filename in os.listdir(IMG_PATH):

        img = cv2.imread(os.path.join(IMG_PATH,filename))
        if(img is not None):
            images[filename] = img
        else:            
            print("Error with file", filename)
    return images

def main():
    #Read images
    images = read_images()

    for name, img in images.items():

        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = iterative_gaussian_blur(gray_scale,16)
        
        #Otsu 
        _, otsu_thresholded_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
        
        #OPENING
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Perform erosion
        eroded_image = cv2.erode(otsu_thresholded_img, kernel, iterations=30)

        # Perform dilation
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=30)
        
            
        #scale
        circles = cv2.HoughCircles(dilated_image, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                param1=10, param2=10, minRadius=10, maxRadius=3000)

        if circles is not None:
            # Calculate average size of detected circles
            average_radius = np.mean(circles[0, :, 2])

            # Choose a reference diameter 
            reference_diameter = 100

            # Calculate scale factor
            scale_factor = reference_diameter / (2 * average_radius)

            # Resize the image using the scale factor
            resized_bin_image = cv2.resize(dilated_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            
            resized_image = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            #Circle 
            if(resized_bin_image is not None):
                cimg,coins = findCoin(resized_bin_image)

                cv2.imwrite(name,cimg)
                
                if coins is not None:
                    myClassifier.coin_finder_accuracy(name,len(coins[0]))
                    amount = myClassifier.findValue(resized_image,coins)
                    myClassifier.compute_amount_MAE(name, amount)
                else:
                    print("(Reshaped) No coin found : ",name)
                    myClassifier.coin_finder_accuracy(name,0)

                    #myClassifier.print_coin_finder_accuracy(name,0)

        else:
                cimg,coins = findCoin(dilated_image)
                
                cv2.imwrite(name,cimg)

                if coins is not None:
                    myClassifier.coin_finder_accuracy(name,len(coins[0]))
                    amount = myClassifier.findValue(resized_image,coins)
                    myClassifier.compute_amount_MAE(name, amount)
                else:
                    print("No coin found : ",name)
                    myClassifier.coin_finder_accuracy(name,0)

                    #myClassifier.print_coin_finder_accuracy(name,0)
                    


    myClassifier.print_coin_finder_evaluation()
    myClassifier.print_overall_mae()
            

            

    
if __name__ == "__main__":
    main()


# BEST RESULT:
#1)
#    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 40,
#                        param1=50, param2=22, minRadius=15, maxRadius=1000)
#Mean coin finder accuracy: 0.8870866008098708
#Overall MAE for amount :  55.00768292682929
#2)
#Mean coin finder accuracy: 0.7999374502621317
#Overall MAE for amount :  42.78109756097563