# -*- coding: utf-8 -*-
"""
Dataset loader script for Preprocessed Simpsons dataset as initial script to run for main programm
This scripts loads the dataset object, divides this into train & validate.
Furthermore the datasets will be preprocessed by scaling into range (0,1) in order to fasten up training for the CNN.

@author: Tuan Le
@email: tuanle@hotmail.de
"""

### Load libraries and needed modules:
from os.path import dirname
import numpy as np
import h5py
#from keras.utils import np_utils

#Define location of datasets
dataset_path = f"{dirname(__file__)}/../data/dataset.h5"
label_path = f"{dirname(__file__)}/../data/labels.h5"

##Testing:
#dataset_path = "C:/Users/tuan.le/Desktop/DeepLearning/Simpsons/data/dataset.h5"
#label_path = "C:/Users/tuan.le/Desktop/DeepLearning/Simpsons/data/labels.h5"
##
def load_data():
    """
    Data loader  and preprocessor for Simpsons colour images for CNN

    Returns
    ---------
    X_train : numpy.ndarray
        Scaled train feature data for CNN
    X_test: numpy.ndarray
        Scaled test feature data  for CNN
    y_train : numpy.ndarray
        Converted training target as one-hot encoded matrix for each class
    y_test : numpy.ndarray
        Converted test target as one-hot encoded matrix for each class
    """ 
    # Load dataset which contains X_train and X_test
    h5f = h5py.File(dataset_path,"r+")
    X_train = h5f["X_train"][:]
    print("Shape of training set:", X_train.shape)
    X_test = h5f["X_test"][:]
    print("Shape of test/validation set:", X_test.shape)
    h5f.close()    

    # Load labels for X_train--> y_train and X_test--> y_test
    h5f = h5py.File(label_path,"r+")
    y_train = h5f["y_train"][:]
    print("Number of classes:", y_train.shape[1])
    y_test = h5f["y_test"][:]
    h5f.close()  
    
    # Scale X_data into range (0-1)   
    np.max(X_train)
    X_train = X_train.astype("float32") / np.max(X_train)
    X_test = X_test.astype("float32") / np.max(X_test)
    
    print("Validation-Train Ratio:", X_test.shape[0]/X_train.shape[0])
    
    return X_train, y_train, X_test, y_test
