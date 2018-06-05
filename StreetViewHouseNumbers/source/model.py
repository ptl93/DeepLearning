# -*- coding: utf-8 -*-
"""
Convolutional Neural Network model setup
This scripts configures a convolutional neural networking using keras and tensorflow backend
http://ufldl.stanford.edu/housenumbers/
@author: Tuan Le
@email: tuanle@hotmail.de
"""

#Import needed libraries and modules:
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

### cnn params ###

color_channels = 3
image_size = 32
n_classes = 11

# 1st layer:
depth_1 = 64
filter_1 = 3

# 2nd layer:
depth_2 = 48
filter_2 = 3
pool_2 = 2
dropout_2 = 0.2

# 3rd layer:
depth_3 = 32
filter_3 = 3
pool_3 = 2
dropout_3 = 0.25

## flatten nn
flatten_size_1 = 512
dropout_flatten_1 = 0.3
flatten_size_2 = 256
dropout_flatten_2 = 0.3


#CNN network for Street View House Numbers
def SVHN_CNN(input_shape = (image_size, image_size, color_channels), classes = n_classes):
    model = Sequential()
    
    # 1st conv layer
    model.add(Conv2D(
        depth_1, (filter_1, filter_1), padding = "same", input_shape = input_shape,
        activation= "selu", kernel_initializer = "lecun_normal"
    ))
    model.add(BatchNormalization())
    
    # 2nd conv layer
    model.add(Conv2D(
        depth_2, (filter_2, filter_2), padding = "same", input_shape = input_shape,
        activation = "relu"
    ))
    model.add(MaxPooling2D(pool_size = (pool_2, pool_2)))
    model.add(Dropout(dropout_2))
    
    # 3rd conv layer
    model.add(Conv2D(
        depth_3, (filter_3, filter_3), padding = "same",
        activation = "relu"
    ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_3))
        
    # flat CNN to neural net
    model.add(Flatten())
    model.add(Dense(flatten_size_1, activation = "relu", kernel_initializer = "lecun_normal"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_flatten_1))
    
    model.add(Dense(flatten_size_2, activation = "selu", kernel_initializer = "lecun_normal"))
    model.add(Dropout(dropout_flatten_2))

    # output Layer
    model.add(Dense(classes, activation = "softmax"))
    
    return model