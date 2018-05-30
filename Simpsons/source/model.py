# -*- coding: utf-8 -*-
"""
Convolutional Neural Network model setup
This scripts configures a convolutional neural networking using keras and tensorflow backend

@author: Tuan Le
@email: tuanle@hotmail.de
"""

#Import needed libraries and modules:
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

#CNN network for grayscale Fashion-MNIST
def simpsons_CNN(input_shape = (64, 64, 3), classes = 18):
    model = Sequential()
    
    # 1st conv layer
    model.add(Conv2D(
        64, (3, 3), padding = "same", input_shape = input_shape,
        activation= "selu", kernel_initializer = "lecun_normal"
    ))
    model.add(BatchNormalization())
    
    # 2nd conv layer
    model.add(Conv2D(
        48, (3, 3), padding = "same", input_shape = input_shape,
        activation = "selu", kernel_initializer = "lecun_normal"
    ))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    # 3rd conv layer
    model.add(Conv2D(
        40, (3, 3), padding = "same",
        kernel_initializer = "lecun_normal",
        activation = "selu"
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # 4th conv layer
    model.add(Conv2D(
        64, (3, 3), padding = "same",
        kernel_initializer = "lecun_normal",
        activation = "selu"
    ))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
        
    # flat CNN to neural net
    model.add(Flatten())
    model.add(Dense(512, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(Dropout(0.25))

    # output Layer
    model.add(Dense(classes, activation = "softmax"))
    
    return model