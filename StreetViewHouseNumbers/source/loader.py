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
from sklearn.model_selection import train_test_split
from math import ceil
#Define location of datasets
train_data = f"{dirname(__file__)}/../data/train_32x32.mat"
test_data = f"{dirname(__file__)}/../data/test_32x32.mat"
extra_data = f"{dirname(__file__)}/../data/extra_32x32.mat"

#Testing:
#train_data = "C:/Users/tuan.le/Desktop/DeepLearning/StreetViewHouseNumbers/data/train_32x32.mat"
#test_data = "C:/Users/tuan.le/Desktop/DeepLearning/StreetViewHouseNumbers/data/test_32x32.mat"
#extra_data = "C:/Users/tuan.le/Desktop/DeepLearning/StreetViewHouseNumbers/data/extra_32x32.mat"

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
    ## Get y_train
    y_train = train["y"]
    ## Load Test Set
    test = sio.loadmat(test_data)
    ## From dictionary train extract the X and y components
    X_test = test["X"]
    # Get y_test
    y_test = test["y"]
    del train, test    
    #if only train and test: 
    #train 73257 observations, test 26032 observation: test-train ratio approc: 0.355 -- better use 0.15
    ## train_test_split problematic
    ## stratification might be a problem. lets run it first and see... 
    if load_extra_data == False: 
        X = np.concatenate((X_train, X_test), axis = 3)
        #scale into range (0,1)
        X = X / np.max(X)
        y = np.concatenate((y_train, y_test), axis = 0)
        y = np_utils.to_categorical(y)
        del X_train, X_test
        del y_train, y_test
        n_obs = X.shape[3]
        test_size = ceil(n_obs * 0.15)
        test_idx = np.random.choice(n_obs, size = test_size, replace = False)
        train_idx = np.setdiff1d(np.arange(n_obs), test_idx, assume_unique = True)
        X_train = X[:,:,:,train_idx]
        y_train = y[train_idx,:]
        X_test = X[:,:,:,test_idx]
        y_test = y[test_idx,:]
        del X, y
        
        train_processed = (X_train, y_train)
        print("Shape from X_train:", X_train.shape)
        print("Number of target classes:", y_train.shape[1])
        test_processed = (X_test, y_test)
        print("Shape from X_test:", X_test.shape)
        print("Number of target classes:", y_test.shape[1])
        
    # Load Extra dataset as additional training data if necessary
    if load_extra_data == True: ##might get problematic because of working memory
        extra = sio.loadmat(extra_data)
        ## From dictionary train extract the X and y components
        X_extra = extra["X"]
        ## get y_extra
        y_extra = extra["y"]
        del extra
        X = np.concatenate((X_train, X_test, X_extra), axis = 3)
        del X_train, X_test, X_extra
        y = np.concatenate((y_train, y_test, y_extra), axis = 0)
        del y_train, y_test, y_extra
        y = np_utils.to_categorical(y)
        
        n_obs = X.shape[3]
        test_size = ceil(n_obs * 0.15)
        test_idx = np.random.choice(n_obs, size = test_size, replace = False)
        train_idx = np.setdiff1d(np.arange(n_obs), test_idx, assume_unique = True)
        # scale into range (0,1)
        X = X / np.max(X) 
        ## the division by 255 is problematic. always crashes!
        X_train = X[:,:,:,train_idx]
        y_train = y[train_idx,:]
        X_test = X[:,:,:,test_idx]
        y_test = y[test_idx,:]
        del X, y
        
        train_processed = (X_train, y_train)
        print("Shape from X_train:", X_train.shape)
        print("Number of target classes:", y_train.shape[1])
        test_processed = (X_test, y_test)
        print("Shape from X_test:", X_test.shape)
        print("Number of target classes:", y_test.shape[1])
        
    return train_processed, test_processed
