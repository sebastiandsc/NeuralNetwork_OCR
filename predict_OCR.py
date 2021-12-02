# -*- coding: utf-8 -*-
"""
ICOM 6015
@author: Sebastian Sanchez
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from keras_preprocessing.image import img_to_array


def find_contours(dimensions, img) :

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    img_res = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX)

            char_copy = np.zeros((44,24))
            char = img[intY-2:intY+intHeight+2, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)
            
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res

def segment_characters(image) :

    # Preprocess license plate image
    #img_lp = cv2.resize(image, (333, 70))
    
    img_gray_lp = cv2.resize(image, (333, 80))
    #img_gray_lp = img_lp
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    #plt.imshow(img_binary_lp, cmap='gray')
    #plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    char_list = find_contours(dimensions, img_binary_lp)

    return char_list




#-------------------------------------------------------------------------------------------
# Predicting the output

model = './Model/OCR_Model.h5'
weights = './Model/OCR_weights.h5'
model = load_model(model)  
model.load_weights(weights)

def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    output = ''
    
    for i,ch in enumerate(char):
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(28,28,3)
        plt.figure()
        plt.imshow(img)
        x = img_to_array(img)
        x = np.expand_dims(img,axis=0)
        
        predict = model.predict(x)
        character = predict[0]
        index = int(np.argmax(character))
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        output += characters[index]
    
    return output

#En caso tal se desee obtener la placa desde una imagen de un vehiculo:
#from ObtainPlate import imagenPlate

#En caso tal se desee introducir la imagen de una placa vehicular directamente:
file = 'C:/Users/Sebastian/Desktop/FINALPROJECT_ANN/TEST_licensePlate/LicensePlate (6).jpg'
imagenPlate = cv2.imread(file)
imagenPlate = cv2.cvtColor(imagenPlate, cv2.COLOR_BGR2GRAY)

char = segment_characters(imagenPlate)
print(show_results())