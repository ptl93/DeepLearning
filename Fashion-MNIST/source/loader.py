# -*- coding: utf-8 -*-
"""
Dataset loader script for Fashion-MNIST as initial script to run for main programm
This scripts loads the train data, divides this into train & validate and loads the test data for submission.
Furthermore the datasets will be preprocessed in order to fasten up training for the CNN.

@author: Tuan Le
@email: tuanle@hotmail.de
"""

### Load libraries and needed modules:
from os.path import dirname
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

#Define location of datasets
data_train_path = f"{dirname(__file__)}/../data/train.csv"
data_submission_path = f"{dirname(__file__)}/../data/test.csv"

def load_data(validate_size = 0.2, random_seed = 26):
    """
    Data loader  and preprocessor for grayscale images for CNN
    
    Parameter
    ---------
    data_train_path: string
        full path for train.csv to be splitted into train and validate
    data_submission_path: string
        full path for submission data
    validate_size : float
        fraction in (0,1) for validation size. Default: 0.2
    random_seed: integer
        Random seed for reproduction
    
    Returns
    ---------
    X_train : numpy.ndarray
        Scaled train feature data with added dimension for CNN
    X_test: numpy.ndarray
        Scaled test feature data with added dimension for CNN
    y_train : numpy.ndarray
        Converted training target with vecor with class labels as categorical vector
    y_test : numpy.ndarray
        Converted test target with vecor with class labels as categorical vector
    """   

    # load data
    data = pd.read_csv(
        data_train_path,
        delimiter = ',', dtype = '|U',
        quotechar = '"', encoding = 'utf-8'
    ).values
    # get X-Features (pixels)
    X = data[:, 1:].astype(dtype = float)
    # get target y-vector
    y = data[:, 0].astype(dtype = float)
    # apply holdout sampling
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size = validate_size, random_state = random_seed
    )
    
    # load test (submission) data
    data_submission = pd.read_csv(
        data_submission_path,
        delimiter = ',', dtype = '|U',
        quotechar = '"', encoding = 'utf-8'
    ).values
    X_submission = data_submission[:, 1:].astype(dtype = float)
    X_submission_id = data_submission[:, 0].astype(dtype = float)
    
    # preprocessing. Scale to(0,1)
    X_train = X_train / max(np.unique(X_train)) 
    X_val   = X_val / max(np.unique(X_val)) 
    X_submission  = X_submission / max(np.unique(X_submission)) 
    
    # 1-0 Encoding for class labels:
    y_train = np_utils.to_categorical(y_train, num_classes = len(np.unique(y_train)))
    y_val   = np_utils.to_categorical(y_val, num_classes = len(np.unique(y_val)))
    
    # enhance additional dimension for convolutional neural net
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_val   = X_val.reshape(X_val.shape[0], 28, 28, 1)
    X_submission  = X_submission.reshape(X_submission.shape[0], 28, 28, 1)
    
    return (X_train, y_train), (X_val, y_val), (X_submission, X_submission_id)
