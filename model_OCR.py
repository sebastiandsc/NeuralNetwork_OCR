# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:21:36 2021

@author: Sebastian
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K

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

imagen_entrenamiento = Train_dataGen.flow_from_directory(
    Train_Data,
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical'
    )

imagen_validacion = Val_dataGen.flow_from_directory(
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

Model.add(Dense(256, activation = 'tanh'))
Model.add(Dropout(0.4))
Model.add(Dense(256, activation='relu'))
Model.add(Dropout(rate = 0.4))
Model.add(Dense(256, activation='relu'))
Model.add(Dropout(rate = 0.4))

Model.add(Dense(classes, activation='softmax'))

Model.summary()


Model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001),metrics=('accuracy'))

History = Model.fit_generator(imagen_entrenamiento, epochs = 80, 
                    steps_per_epoch = 864//32, 
                    validation_data=imagen_validacion)

dir = 'C:/Users/Sebastian/Desktop/REDES NEURONALES'
if not os.path.exists(dir):
    os.mkdir(dir)
Model.save('./modelo/OCR_Model.h5')
Model.save_weights('./modelo/OCR_weights.h5')
