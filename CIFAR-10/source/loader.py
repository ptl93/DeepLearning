# -*- coding: utf-8 -*-
"""
Dataset loader script for CIFAR-10 dataset as initial script to run for main programm
This scripts loads the dataset object, divides this into train & test.
X-Data will be resized into 64x64x3 
Furthermore the datasets will be preprocessed by standardizing [ x - mean(x) / sd(x)] in order to fasten up training for the CNN.
## To download whole data dictionary go to : http://www.cs.utoronto.ca/~kriz/cifar.html
@author: Tuan Le
@email: tuanle@hotmail.de
"""

### Load libraries and needed modules:
from os.path import dirname
import numpy as np
from keras.utils import np_utils
import tarfile, sys
import pickle
from scipy.misc import imresize

#Define location of data
data_dir = f"{dirname(__file__)}/../data/"
#local testing
#data_dir = "C:/Users/tuan.le/Desktop/DeepLearning/CIFAR-10/data/"

#for info like train and test size
info_dir = f"{dirname(__file__)}/../model/info.txt"
#info_dir = "C:/Users/tuan.le/Desktop/DeepLearning/CIFAR-10//model/info.txt"

# function for extracting tar.gz archive
def untar(data_dir):
    fname = data_dir + "cifar-10-python.tar.gz"
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall(path = data_dir)
        tar.close()
        print("Extracted in Data Directory:", data_dir)
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])
 
# batch reader
def unpickle(file):
    with open (file, "rb") as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

# reshape data such that into rgb
def cifar_10_reshape(batch_arg):
    output = np.reshape(batch_arg,(10000,3,32,32)).transpose(0,2,3,1)
    return output

# resize data images to 64x64 for better resolution
def resize(images, height = 64, width = 64, color_channels = 3) :
    X = np.zeros((images.shape[0], height, width,3))
    for i in range(images.shape[0]):
        X[i]= imresize(images[i], (height,width, color_channels), interp = 'bilinear', mode = None)
    return X

def load_data():
    
    """
    Data loader  and preprocessor for SHVN colour images for CNN
    
    Returns
    ---------
    train_processed: tuple of 2 numpy array for resized X_train (standardized) and y_train (1-0 hot encoded mat)
    test_processed: tuple of 2 numpy array for resized X_test (standardized) and y_test (1-0 hot encoded mat)
    """
    
    ## Untar the archived files
    untar(data_dir)
    ## Define new path where unzipped files lie
    data_dir = data_dir + "cifar-10-batches-py"
    ## Define training batches:
    training_batches = ["data_batch_" + str(i) for i in range(1, 6)]
    ## Load all training batches directly and concatenate
    for i in range(0, 5):
        batch_full_path = data_dir + "/" + training_batches[i]
        tmp = unpickle(batch_full_path)
        if i == 0: #Init 
            X_train = cifar_10_reshape(tmp[list(tmp.keys())[2]])
            y_train = tmp[list(tmp.keys())[1]]
        else: #Concat
            X_train = np.concatenate((X_train, cifar_10_reshape(tmp[list(tmp.keys())[2]])), axis = 0)
            y_train = np.concatenate((y_train, tmp[list(tmp.keys())[1]]))

    ## Load test batch
    batch_full_path_test = data_dir + "/test_batch"
    tmp = unpickle(batch_full_path_test)
    X_test = cifar_10_reshape(tmp[list(tmp.keys())[2]])
    y_test = np.array(tmp[list(tmp.keys())[1]])
    
    ## Preprocess:
    ## Train
    X_train = X_train.astype('float32')
    print("Shape of training set:", X_train.shape)
    print("Resizing training set images:")
    X_train = resize(X_train)
    X_test = X_test.astype('float32')
    print("Shape of resized test set:", X_test.shape)
    ## Test
    X_test = X_test.astype('float32')
    print("Shape of training set:", X_train.shape)
    print("Resizing test set images:")
    X_test = resize(X_test)
    print("Shape of resized test set:", X_test.shape)
    print("Now standardizing X_train and X_test, assuming all images come from one 'image generating' distribution.")
    mean = np.mean(X_train, axis = (0,1,2,3))
    std = np.std(X_train, axis = (0,1,2,3))
    X_train = (X_train - mean)/(std + 1e-7)
    X_test = (X_test - mean)/(std + 1e-7)
    
    ## One-Hot Encoding for targets
    print("Convert target vectors for train and test into matrices as one-hot-encode")
    y_train = np_utils.to_categorical(y_train, num_classes = len(np.unique(y_train)))
    y_test = np_utils.to_categorical(y_test, num_classes = len(np.unique(y_test)))
    
    train_processed = (X_train, y_train)
    test_processed = (X_test, y_train)
    
    #write train and test info into ../model/info.txt 
    info_file = open(info_dir, "w")
    train_info = "Training info shape: " + str(X_train.shape)
    test_info = "Test info shape: " + str(X_test.shape)
    info_file.write(train_info)
    info_file.write("\n")
    info_file.write(test_info)
    info_file.close()
    
    return train_processed, test_processed
