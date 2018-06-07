# -*- coding: utf-8 -*-
"""
Main python programm for classifying Fasion-MNIST dataset

@author: Tuan Le
@email: tuanle@hotmail.de
"""
from loader import load_data
from model import fashion_CNN
import json
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

# define hyperparameter class (object)
class Hyperparams:
    def __init__(self):
        self.pretrain = False #True 
        self.fullset = True
        self.architecture = "../model/architecture.json"
        self.inter_model = "../model/intermediate.h5"
        self.submission  = "../data/submission.csv"
        self.history = "../model/history.json"
        # training params
        self.batch_size = 128
        self.epochs = 50
        self.input_shape = (28, 28, 1)
        self.classes = 10

# python main programm function
def main():
    # init Hyperparams object
    params = Hyperparams()
    
    # load preprocessed data
    (X_train, y_train), (X_val, y_val), (X_submission, X_submission_id) = load_data()
    
    # init CNN object
    model = fashion_CNN()
    # print model summary to console
    print(model.summary())
    # if trained weights already exist
    if params.pretrain:
        model.load_weights(params.inter_model)
        model.compile(
            loss = categorical_crossentropy,
            optimizer = adam(lr = 1e-3),
            metrics = ["accuracy"]
        )
    # train CNN from start
    else:
        generator_train = ImageDataGenerator(
            rotation_range = 30,
            shear_range = 0.3,
            width_shift_range = 0.08,
            height_shift_range = 0.08,
            zoom_range = 0.08,
            horizontal_flip = True,
            vertical_flip = True,
        ).flow(X_train, y_train, batch_size = params.batch_size)
        # train history
        callbacks = [
            ModelCheckpoint(
                filepath = params.inter_model, monitor = "val_acc",
                save_best_only = True, verbose=1, mode = "max"
            )
        ]
        # configure and compile CNN with adam algorithm
        model.compile(
            loss = categorical_crossentropy,
            optimizer = adam(lr = 1e-3),
            metrics = ["accuracy"]
        )
        history = model.fit_generator(
            generator_train,
            X_train.shape[0] // params.batch_size,
            epochs = params.epochs,
            validation_data = (X_val, y_val),
            validation_steps = (X_val.shape[0] // params.epochs),
            callbacks = callbacks
        )

    # model performance
    train_results = model.evaluate(X_train, y_train)
    val_results = model.evaluate(X_val, y_val)
    print("Train       Loss: ", train_results[0])
    print("Train       Acc: ",  train_results[1])
    print("Validation  Loss:",  val_results[0])
    print("Validation  Acc: ",  val_results[1])
    # saving archicteture:
    with open(params.architecture, 'w') as fp:
        json.dump(model.to_json(), fp)
    # saving history
    if params.pretrain == False:
        with open(params.history, "w") as jp:
            json.dump(history.history, indent = 4, separators=(',', ': '), fp = jp)
    # saving prediction
    y_pred = model.predict(X_submission)
    y_pred = np.argmax(y_pred, axis = 1)
    X_submission_id = X_submission_id.reshape(X_submission_id.shape[0], 1)
    y_pred = y_pred.reshape(y_pred.shape[0], 1)
    results = np.concatenate((X_submission_id, y_pred), axis = 1)
    # save submission y predicted labels
    np.savetxt(params.submission, results, '%d', delimiter = ',')
    

# run main program when calling main.py programm in shell
if __name__ == '__main__':
    main()