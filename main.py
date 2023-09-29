import os
import random
from typing import List

import pandas as pd
import tensorflow as tf
import numpy as np
import keras as k
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
# Define the path to your dataset
path_folder = r"C:\Users\zbq46b\Desktop\Bc_Work\testPlants"

# List the directories inside the dataset folder and sort them
plant_dataset = os.listdir(path_folder)
plant_dataset.sort()
plant_dataset: list[str] = plant_dataset[:6] + plant_dataset[20:20]
# Define the class names for your dataset
class_names = ['Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust',
               'Apple_Powdery_mildew', 'Apple_healthy']

print(class_names)

image_data = []  # Create an empty list to store images
label_data = []  # Create an empty list to store labels

count = 0

for folder in plant_dataset:
    images = os.listdir(path_folder + "/" + folder)
    print("Loading Folder -- {} ".format(folder), "The Count of Classes ==> ", count)

    for img in images:
        image = cv2.imread(path_folder + "/" + folder + "/" + img)
        if image is not None:
            image = cv2.resize(image, (100, 100))
            image_data.append(image)  # takes an object as an argument and adds it to the end of an existing list
            label_data.append(count)  # Add the label (class index) to label_data
    count += 1
        #print("Folder name:", folder)
print("---- Done ----------- ")
data = np.array(image_data)  # converts the image_data list into a NumPy array called data.
data = data.astype('float32')  # converts data to suitable data format for ML
data = data/255.0  # divide data by 255, image is in range 0 to 255
# print(data.shape)
label = np.array(label_data)
label_num = to_categorical(label, len(plant_dataset))
#label_num[100]
label_num = label_num.astype('int32')
# to_categorical = encodes the class labels
print(label)
print("===========================LABEL=======================")
print(label_data[0])
# print(len(image_data)) all images are stored successfully
print("===========================DATA=======================")
print(data.shape[3])
print("====================LABEL==============================")
# print(label_num)
x_img, y_img = shuffle(data, label_num)
x_train, x_test, y_train, y_test = train_test_split(x_img, y_img, train_size=0.8)

print(x_train.shape, x_test.shape, y_test.shape, y_train.shape)
