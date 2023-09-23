import os
from typing import List

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

import keras as k
# Define the path to your dataset
path_folder = r"C:\Users\zbq46b\Desktop\archive\MyDataset"

# List the directories inside the dataset folder and sort them
plant_dataset = os.listdir(path_folder)
plant_dataset.sort()
plant_dataset: list[str] = plant_dataset[:4] + plant_dataset[20:20]
# Define the class names for your dataset
class_names = ['apple_healthy', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_Powdery_mildew', 'Apple_scab']

print(path_folder)

image_data = []  # Create an empty list to store images
label_data = []  # Create an empty list to store labels

count = 0

for folder in plant_dataset:
    images = os.listdir(path_folder + "/" + folder)
    print("Loading Folder -- {} ".format(folder), "The Count of Classes ==> ", plant_dataset.count(folder))
    for img in images:
        image = cv2.imread(path_folder + "/" + folder + "/" + img)
        if image is not None:
            image = cv2.resize(image, (100, 100))
            image_data.append(image)
            label_data.append(class_names.index(folder))  # Add the label (class index) to label_data
        count += 1
        print("Folder name:", folder)
print("---- Done ----------- ")
