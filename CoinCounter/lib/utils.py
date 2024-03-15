import pandas as pd
import cv2
import os

DIRNAME = os.path.dirname(os.getcwd())
IMG_PATH = os.path.join(DIRNAME,"data")

def clean_dataset(dataset):
    dataset = dataset.rename(columns={'Nom image': 'image_label', 'Nombre de pièces': 'nb_coins', 'Valeur ': 'value'})
    split_values = dataset['value'].str.split().tolist()
    values_dict_list = []
    for value in split_values:
        if len(value) > 2 :
            values_dict_list.append({"units":value[0],"cents":value[2]})
        elif len(value) == 2 :
            if "euro" in value[1]:
                values_dict_list.append({"units":value[0],"cents":0})
            elif "cent" in value[1]:
                values_dict_list.append({"units":0,"cents":value[0]})
    values_df = pd.DataFrame(values_dict_list)
    dataset = pd.concat([dataset, values_df], axis=1)
    return dataset

def read_images():
    images = {}
    for filename in os.listdir(IMG_PATH):
        img = cv2.imread(os.path.join(IMG_PATH, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename] = img
        else:
            print("Error with file", filename)
    return images
