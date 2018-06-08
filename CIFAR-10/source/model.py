# -*- coding: utf-8 -*-
"""
Convolutional Neural Network model setup
This scripts configures a convolutional neural networking using keras and tensorflow backend
http://www.cs.utoronto.ca/~kriz/cifar.html

The CNN-Model is a modification of VGG16 Neural Network
Modifications: Less convolutional layers due to training time, BatchNormalization...
See paper https://arxiv.org/abs/1409.1556
@author: Tuan Le
@email: tuanle@hotmail.de
"""

#Import needed libraries and modules:
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.regularizers import l2

#CNN network for CIFAR-10
image_size = 64
color_channels = 3
n_classes = 10

## Simplified CNN for training:
def cifar10_CNN(input_shape = (image_size, image_size, color_channels), classes = n_classes):
    model = Sequential()
    
    ##################
    # 1st conv layer
    model.add(Conv2D(
        64, (3, 3), padding = "same", input_shape = input_shape,
        activation= "relu", kernel_regularizer = l2(1e-4)))
    # 2nd conv layer
    model.add(Conv2D(
        48, (3, 3), padding = "same", activation= "relu",
        kernel_regularizer = l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    # max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    ##################
    
    ##################
    # 3nd conv layer
    model.add(Conv2D(
        128, (3, 3), padding = "same", activation= "relu",
        kernel_regularizer = l2(1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # 4th conv layer
    model.add(Conv2D(
        64, (3, 3), padding = "same", activation = "relu",
        kernel_regularizer = l2(1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    ##################
    
    ##################
    model.add(Conv2D(
            32, (3, 3), activation='relu', padding='same',
            kernel_regularizer = l2(1e-5)))
    model.add(Conv2D(
            32, (3, 3), activation='relu', padding='same',
            kernel_regularizer = l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   
    ##################
    
    # flat CNN to neural net
    model.add(Flatten())
    model.add(Dense(512, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(256, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # output Layer
    model.add(Dense(classes, activation = "softmax"))
    
    return model

## Modified VGG Net
def cifar10_CNN_VGG_modified(input_shape = (image_size, image_size, color_channels), classes = n_classes):
    model = Sequential()
    
    ##################
    # 1st conv layer
    model.add(Conv2D(
        64, (3, 3), padding = "same", input_shape = input_shape,
        activation= "relu", kernel_regularizer = l2(1e-4)))
    # 2nd conv layer
    model.add(Conv2D(
        64, (3, 3), padding = "same", activation= "relu",
        kernel_regularizer = l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    # max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    ##################
    
    ##################
    # 3nd conv layer
    model.add(Conv2D(
        128, (3, 3), padding = "same", activation= "relu",
        kernel_regularizer = l2(1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # 4th conv layer
    model.add(Conv2D(
        128, (3, 3), padding = "same", activation = "relu",
        kernel_regularizer = l2(1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    ##################
    
    ##################
    model.add(Conv2D(
            256, (3, 3), activation='relu', padding='same',
            kernel_regularizer = l2(1e-5)))
    model.add(Conv2D(
            256, (3, 3), activation='relu', padding='same',
            kernel_regularizer = l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   
    ##################
    
    # flat CNN to neural net
    model.add(Flatten())
    model.add(Dense(512, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(Dense(256, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(128, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # output Layer
    model.add(Dense(classes, activation = "softmax"))
    
    return model

## Real VGG NET
    def VGG16(input_shape = (image_size, image_size, color_channels), classes = n_classes):
        model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
               activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(n_classes, activation='softmax')
        ])
        return model