import pandas as pd
import numpy as np
import cv2
class Classifier():
    
    COIN_VALUES = {
        "nb_two": 2,
        "nb_one": 1,
        "nb_ten": 0.10,
        "nb_twenty": 0.20,
        "nb_fifty": 0.50,
        "nb_five": 0.05,
        "nb_zero_two": 0.02,
        "nb_zero_one": 0.01
    }
    
    def __init__(self):
        self.label = self.read_csv()
        self.amount_MAE = 0
        self.nb_images = len(self.label)
        self.true_pos = 0
        self.false_neg = 0
        self.false_pos = 0
    
    def read_csv(self):
        """
        Method to read CSV file containing image labels.
        
        Returns:
            DataFrame: Pandas DataFrame containing image labels.
        """
        LABEL_PATH = "/Users/nour/Documents/M1/imgProcessing/CoinCounter/data_clean.csv"
        return pd.read_csv(LABEL_PATH)

    def coin_finder_accuracy(self, img_name,found):
        """
        Evaluate the accuracy of coin detection for a given image.
        
        Args:
            img_name (str): Name of the image.
            found (int): Number of coins found in the image.
        """
        try:
            truth = self.label.loc[self.label['image_label'] == img_name, 'nb_coins'].values[0]

            if(found > truth):
                self.false_pos += (found-truth)
                self.true_pos += truth
                
            else:
                self.true_pos += found
                self.false_neg += (truth-found)

        except:
            print("Error with: ",img_name)

    def print_coin_finder_evaluation(self):
        """
        Print precision and recall for coin detection.
        """
        precision = self.true_pos / (self.true_pos + self.false_pos)
        recall = self.true_pos / (self.true_pos + self.false_neg)
        print("Precision: ", precision)
        print("Recall: ", recall)

    def compute_amount_MAE(self,img_name,found_amount):
        """
        Compute mean absolute error (MAE) for detected amount in a given image.
        
        Args:
            img_name (str): Name of the image.
            found_amount (float): Detected amount of money in the image.
        """
        try:
            unit = self.label.loc[self.label['image_label'] == img_name, 'units'].values[0]
            cent = self.label.loc[self.label['image_label'] == img_name, 'cents'].values[0]
            total = unit * 1 + cent * 0.01

            mae = abs(total - found_amount)
            self.amount_MAE += mae
            print("MAE for image " + img_name + " : "+ str(mae))
        except:
            print("Error with: ",img_name)
    
    def print_overall_mae(self):
        """
        Print overall mean absolute error (MAE) for amount.
        """
        print("Overall MAE for amount : ", str(self.amount_MAE/self.nb_images))
    
    def draw_coin_value(self,image,value,text,coordinate):
        """
        Draw coin value on the image.
        
        Args:
            image (numpy.ndarray): Input image.
            value (float): Coin value.
            text (str): Text to be displayed.
            coordinate (tuple): Coordinate to place the text.
        """
        font_scale = min(image.shape[0], image.shape[1]) / 1000.0  

        cv2.putText(image, f'{text}: {value}', coordinate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)

        
    def findValue(self,cimg ,image, circles):
        """
        Detect and compute the value of coins in the image.
        
        Args:
            cimg (numpy.ndarray): Color image with detected circles.
            image (numpy.ndarray): Grayscale image.
            circles (numpy.ndarray): Detected circles.
        
        Returns:
            float: Total value of coins detected in the image.
        """
        coin_counts = {
            "nb_two": 0,
            "nb_one": 0,
            "nb_ten": 0,
            "nb_twenty": 0,
            "nb_fifty": 0,
            "nb_five": 0,
            "nb_zero_two": 0,
            "nb_zero_one": 0
        }       
        for circle in circles[0,:]:
            x_center, y_center = circle[0], circle[1]
            radius = circle[2]
            color_at_center = image[y_center, x_center].astype(float)
            diameter = radius**2
            
            #Red Coin delta between G and R big or factor between R and B >2
            if(abs(color_at_center[2] - color_at_center[1]) >=37 or color_at_center[2]/color_at_center[0] > 2.3):
                if(diameter <= 1800):
                    coin_counts["nb_zero_one"] +=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_zero_one"],"Val",(x_center,y_center))
                elif(diameter >1800 and diameter< 2100):
                    coin_counts["nb_zero_two"] +=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_zero_two"],"Val",(x_center,y_center))
                else:
                    coin_counts["nb_five"] +=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_five"],"Val",(x_center,y_center))
            #Silver Coin
            elif(abs(color_at_center[0] - color_at_center[1]) < 25 and abs(color_at_center[0] - color_at_center[2]) <  25):
                    coin_counts["nb_one"]+=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_one"],"Val",(x_center,y_center))
            #GOLD Coin
            else:
                if (diameter <= 3900):
                    coin_counts["nb_ten"] +=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_ten"],"Val",(x_center,y_center))
                elif(diameter > 3900 and diameter <4200):
                    coin_counts["nb_twenty"] +=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_twenty"],"Val",(x_center,y_center))
                elif(diameter > 4800):
                    coin_counts["nb_two"]+=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_two"],"Val",(x_center,y_center))
                else:
                    coin_counts["nb_fifty"] +=1
                    self.draw_coin_value(cimg,self.COIN_VALUES["nb_fifty"],"Val",(x_center,y_center))
                    
        return sum(coin_counts[key] * self.COIN_VALUES[key] for key in coin_counts)

