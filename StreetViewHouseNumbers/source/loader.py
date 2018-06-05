# -*- coding: utf-8 -*-
"""
Dataset loader script for StreetView House Numbers dataset as initial script to run for main programm
This scripts loads the dataset object, divides this into train & validate.
Furthermore the datasets will be preprocessed by scaling into range (0,1) in order to fasten up training for the CNN.
## To download format2 from: http://ufldl.stanford.edu/housenumbers/
@author: Tuan Le
@email: tuanle@hotmail.de
"""

### Load libraries and needed modules:
from os.path import dirname
import numpy as np
import scipy.io as sio
from keras.utils import np_utils

#Define location of datasets
train_data = f"{dirname(__file__)}/../data/train_32x32.mat"
test_data = f"{dirname(__file__)}/../data/test_32x32.mat"
extra_data = f"{dirname(__file__)}/../data/extra_32x32.mat"

#Testing:
train_data = "C:/Users/tuan.le/Desktop/DeepLearning/StreetViewHouseNumbers/data/train_32x32.mat"
test_data = "C:/Users/tuan.le/Desktop/DeepLearning/StreetViewHouseNumbers/data/test_32x32.mat"
extra_data = "C:/Users/tuan.le/Desktop/DeepLearning/StreetViewHouseNumbers/data/extra_32x32.mat"

def load_data(load_extra_data = False):
    
    """
    Data loader  and preprocessor for SHVN colour images for CNN
    
    Returns
    ---------
    train_processed: tuple of 2 numpy array for X_train (scaled into 0,1) and y_train (1-0 hot encoded mat)
    test_processed: tuple of 2 numpy array for X_test (scaled into 0,1) and y_test (1-0 hot encoded mat)
    extra: if load_extra_data = True, same like above for extra data
    """
    ## Load Train Set
    train = sio.loadmat(train_data)
    ## From dictionary train extract the X and y components
    X_train = train["X"]
    ## Scale X_train into range (0,1)
    X_train = X_train / np.max(X_train)
    ## 1-0 Hot encoding into matrix for each class
    y_train = np_utils.to_categorical(train["y"])
    ## Put into tuple
    train_processed = (X_train, y_train)
    print("Shape from X_train:", X_train.shape)
    print("Number of target classes:", y_train.shape[1])
    
    ## Load Test Set
    test = sio.loadmat(test_data)
    ## From dictionary train extract the X and y components
    X_test = test["X"]
    ## Scale X_test into range (0,1)
    X_test = X_test / np.max(X_test)
    #1-0 Hot encoding into matrix for each class
    y_test = np_utils.to_categorical(test["y"])
    ## Put into tuple
    test_processed = (X_test, y_test)
    print("Shape from X_test:", X_test.shape)
    print("Number of target classes:", y_test.shape[1])
    # Load Extra dataset as additional training data if necessary
    if load_extra_data:
        extra = sio.loadmat(train_data)
        ## From dictionary train extract the X and y components
        X_extra = extra["X"]
        ## Scale X_extra into range (0,1)
        X_extra = X_extra / np.max(X_extra)
        ## 1-0 Hot encoding into matrix for each class
        y_extra = np_utils.to_categorical(extra["y"])  
        extra = (X_extra, y_extra)
    else:
        extra = None
    return train_processed, test_processed, extra
