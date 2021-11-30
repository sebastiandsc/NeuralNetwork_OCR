# -*- coding: utf-8 -*-
"""
ICOM 6015
@author: Sebastian Sanchez
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

K.clear_session()
    
Train_Data = "C:/Users/Sebastian/Desktop/REDES NEURONALES/data/train"
Val_data = "C:/Users/Sebastian/Desktop/REDES NEURONALES/data/val"


Train_dataGen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
    )

Val_dataGen = ImageDataGenerator(
    rescale=1./255   
    )

train_data_img = Train_dataGen.flow_from_directory(
    Train_Data,
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical'
    )

val_data_img = Val_dataGen.flow_from_directory(
    Val_data,
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical'
    )

classes = 36

# Model

Model = Sequential()

Model.add(Convolution2D(16,(3,3), 
                        padding='same', 
                        input_shape=(28,28,3),
                        activation='relu'))

Model.add(MaxPooling2D(pool_size=(2,2)))

Model.add(Convolution2D(32, (3,3),
                        padding='same', 
                        activation='relu'))

Model.add(MaxPooling2D(pool_size=(2,2)))

Model.add(Convolution2D(64, (3,3),
                        padding='same', 
                        activation='relu'))

Model.add(MaxPooling2D(pool_size=(2,2)))

Model.add(Flatten())

Model.add(Dense(128, activation = 'tanh'))
Model.add(Dropout(0.4))
Model.add(Dense(256, activation='relu'))
Model.add(Dropout(rate = 0.4))
Model.add(Dense(512, activation='relu'))
Model.add(Dropout(rate = 0.4))
Model.add(Dense(128, activation = 'tanh'))
Model.add(Dropout(0.4))
Model.add(Dense(classes, activation='softmax'))

Model.summary()


Model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001),metrics=('accuracy'))

History = Model.fit_generator(train_data_img, epochs = 120, 
                    steps_per_epoch = 864//32, 
                    validation_data=val_data_img)

scores = Model.evaluate(val_data_img)
print("%s%s: %.2f%%" % ("evaluate_generator ",Model.metrics_names[1], scores[1]*100))

#Confution Matrix and Classification Report
Y_pred = Model.predict(val_data_img, 864//33)
y_pred = np.argmax(Y_pred, axis=1)


Matrix_C = confusion_matrix(val_data_img.classes, y_pred)

target_names = [] 
for x in range(0,36): target_names.append(str(x))

Class_Report = classification_report(val_data_img.classes, y_pred, target_names=target_names)
print(Class_Report)

plt.plot(History.history["loss"], label="train_loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()


dir = 'C:/Users/Sebastian/Desktop/REDES NEURONALES'
if not os.path.exists(dir):
    os.mkdir(dir)
Model.save('./modelo/OCR_Model.h5')
Model.save_weights('./modelo/OCR_weights.h5')