# -*- coding: utf-8 -*-
"""
Easy implementation of a convolutional neural network for the MNIST dataset

Environment:
    Keras 
    TensorFlow (backend)
    NumPy
    Pandas
    h5py
    matplotlib
    
@author: Tuan Le
@email: tuanle@hotmail.de
"""

### Load libraries and modules 
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist

os.chdir("Desktop\\DeepLearning\\MNIST")
os.getcwd()

### Load MNIST data into training and test sets:
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Have a look at the shape
X_train.shape, X_test.shape
#(60000, 28, 28)
#(10000, 28, 28)

#grey scale 0-255:
print(np.unique(X_train))

y_train.shape, y_test.shape
#(60000,)
#(10000,)

### Handwritten digits from 0 to 9, since:
np.unique(y_train)
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)

#Plot images
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Class {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig

### For training 60k observations. Since MNIST are grey-scale image, the images are a 2dimensional matrix of shape 28x28
### Since in this implementation a convolutional neural net is used, the third dimension has to be added, resulting as RGB image

### Preprocessing:
def preprocess(X_train, X_test, y_train, y_test):
    """
    Data preprocessor for grayscale images for CNN
    
    Parameter
    ---------
    X_train : numpy.ndarray
        Training feature data as matrix 
    X_test : numpy.ndarray
        Test feature data as matrix
    y_train : numpy.ndarray
        Training target vector with class labels
    y_test: numpy.ndarray
        Test target vector with class labels
    
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
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) # Leading to 60000  x 28 x 28 x 1
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1) # 10000 x 28 x 28  x 1
    ## Type is currently uint8 because of grayscale 0-255. Convert into float 
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    ## Since images are greyscale with artifically adding the third dimension, for the greyscale part the values have a range from 0-255(single grayscale).
    ## For training scale the values to have range [0,1]
    X_train = X_train/max(np.unique(X_train))
    X_test = X_test/max(np.unique(X_test))
    ## Convert target variable (class) from uint8 into factor/categorical. Here MNIST 10 class labels 0-9
    y_train = np_utils.to_categorical(y_train, num_classes = len(np.unique(y_train)))
    y_test = np_utils.to_categorical(y_test, num_classes = len(np.unique(y_test)))
    ## Return
    return X_train, X_test, y_train, y_test

### Apply preprocessing:   
X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)


### Define convolutional neural net architecture:
## Hyperparams:
pic_size = 28 # image have size 28x28

# convolutional layer 1 - 32 filter, each kernel 3x3
n_conv_1 = 32
k_size_1 = 3
act_1 = "relu"

# convolutional layer 2 - 64 filters, each kernel 3x3
n_conv_2 = 48
k_size_2 = 3
act_2 = "relu"

# max pool layer - Poolong size 2x2 for dimension reduction. Drop 25% of nodes/neurons
p_size_1 = 2
dropout_1 = .25

# dense layer
n_dense_1 = 128
dense_activ = "relu"
dropout_2 = 0.5

num_classes = 10

def conv_net(pic_size, n_conv_1, k_size_1, act_1, n_conv_2, k_size_2, act_2, p_size_1, dropout_1, n_dense_1, dense_activ, dropout_2):
    # Init structure
    model = Sequential()
    # Convolutional layer
    model.add(Conv2D(filters = n_conv_1, kernel_size = (k_size_1, k_size_1), activation = act_1, input_shape=(pic_size, pic_size, 1)))
    model.add(Conv2D(filters = n_conv_2, kernel_size = (k_size_2, k_size_2), activation = act_2))
    model.add(MaxPooling2D(pool_size=(p_size_1,p_size_1)))
    model.add(Dropout(dropout_1))
    # Transform back to "vector" by flattening
    model.add(Flatten())
    model.add(Dense(units = n_dense_1, activation = dense_activ))
    model.add(Dropout(dropout_2))
    # Output layer
    model.add(Dense(units = 10, activation="softmax"))
    return model

conv_model = conv_net(pic_size, n_conv_1, k_size_1, act_1, n_conv_2, k_size_2, act_2, p_size_1, dropout_1, n_dense_1, dense_activ, dropout_2)
conv_model.summary()


### Compile model:
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, History

conv_model.compile(loss = categorical_crossentropy,
                   optimizer = adam(),
                   metrics = ["accuracy"])
conv_model.summary()

### Train the model in X_train:
history = History()
checkpoint = ModelCheckpoint("cnn_model_weights_adam.hdf5", monitor = "val_acc", verbose = 0, save_best_only = True, mode = "max")

conv_model.fit(X_train, y_train,
          batch_size=128,
          epochs = 20,
          verbose = 1,
          validation_data = (X_test, y_test),
          callbacks=[history])

### Result at epoch 20 ###
## loss: 0.0143 - acc: 0.9953 - val_loss: 0.0317 - val_acc: 0.9923 ##

### Save model architecture
cnn_model_architecture = conv_model.to_json()
import json
with open("cnn_architecture.json", "w") as fp:
    json.dump(cnn_model_architecture, fp)

### Apply different optimizer:
from keras.optimizers import rmsprop, sgd
history = History()
checkpoint = ModelCheckpoint("cnn_model_weights_rmsprop.hdf5", monitor = "val_acc", verbose = 0, save_best_only = True, mode = "max")

conv_model.compile(loss = categorical_crossentropy,
                   optimizer = rmsprop( lr = 0.001, rho=0.9, decay=0.05),
                   metrics = ["accuracy"]
                   )

conv_model.fit(X_train, y_train,
          batch_size=128,
          epochs = 20,
          verbose = 1,
          validation_data = (X_test, y_test),
          callbacks=[checkpoint, history]
         )

### Result at epoch 20 ###
## loss: 0.1482 - acc: 0.9575 - val_loss: 0.0901 - val_acc: 0.9718s ##
def writeToJSONFile(path, fileName, data):
    filePathNameWExt = "./" + path + "/" + fileName + ".json"
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)
        
history_json_rmsprop = json.dumps(history.history, indent=4, separators=(',', ': '))
writeToJSONFile("./","callback_history_rmsprop", history_json_rmsprop)

# plotting the accuracies over training epochs for train and test dataset:
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig
