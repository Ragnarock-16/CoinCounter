import pandas as pd
import numpy as np

class Classifier():
    
    COIN_VALUES = {
        "nb_two": 2,
        "nb_one": 1,
        "nb_ten": 10,
        "nb_twenty": 20,
        "nb_fifty": 50,
        "nb_five": 5,
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
        LABEL_PATH = "/Users/nour/Documents/M1/imgProcessing/CoinCounter/data_clean.csv"
        return pd.read_csv(LABEL_PATH)

    def coin_finder_accuracy(self, img_name,found):
        
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
        precision = self.true_pos / (self.true_pos + self.false_pos)
        recall = self.true_pos / (self.true_pos + self.false_neg)
        print("Precision: ", precision)
        print("Recall: ", recall)

    def compute_amount_MAE(self,img_name,found_amount):
        try:
            unit = self.label.loc[self.label['image_label'] == img_name, 'units'].values[0]
            cent = self.label.loc[self.label['image_label'] == img_name, 'cents'].values[0]
            total = unit * 1 + cent * 0.01

            mae = abs(total - found_amount)
            self.amount_MAE += mae
            #print("MAE for image " + img_name + " : "+ str(mae))
        except:
            print("Error with: ",img_name)
    
    def print_overall_mae(self):
        print("Overall MAE for amount : ", str(self.amount_MAE/self.nb_images))
        
    def findValue(self, image, circles):
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
            color_at_center = image[y_center, x_center]
            diameter = radius**2

            if(color_at_center[2]>200):
                if(diameter <= 2300):
                    coin_counts["nb_zero_one"] +=1
                elif(diameter >2300 and diameter< 2700):
                    coin_counts["nb_zero_two"] +=1
                else:
                    coin_counts["nb_five"] +=1
            elif(color_at_center[0] < 70 and color_at_center[2] < 140):
                if(diameter<=3600):
                    coin_counts["nb_two"]+=1
                else:
                    coin_counts["nb_one"]+=1
            else:
                if (diameter <= 3900):
                    coin_counts["nb_ten"] +=1
                elif(diameter > 3900 and diameter <4500):
                    coin_counts["nb_twenty"] +=1

                else:
                    coin_counts["nb_fifty"] +=1

        return sum(coin_counts[key] * self.COIN_VALUES[key] for key in coin_counts)

