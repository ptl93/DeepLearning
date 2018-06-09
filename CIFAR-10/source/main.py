# -*- coding: utf-8 -*-
"""
Main python programm for classifying cifar-!= dataset
http://www.cs.utoronto.ca/~kriz/cifar.html
@author: Tuan Le
@email: tuanle@hotmail.de
"""
from loader import load_data
from model import cifar10_CNN, cifar10_CNN_VGG_modified

from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop
from keras.callbacks import ModelCheckpoint
import json
import matplotlib
import matplotlib.pyplot as plt

# define hyperparameter class (object)
class Hyperparams:
    def __init__(self):
        self.pretrain = True #True 
        self.fullset = True
        self.architecture = "../model/architecture.json"
        self.inter_model = [
            "../model/intermediate_ep60.h5",
            "../model/intermediate_ep90.h5",
            "../model/intermediate_ep120.h5"
        ]
        self.history = [
                "../model/history_epoch60.json",
                "../model/history_epoch90.json",
                "../model/history_epoch120.json"
        ]
        # training params
        self.batch_size = 128
        self.epochs = 30
        self.input_shape = (64, 64, 3)
        self.classes = 10

# python main programm function
def main():
    # init Hyperparams object
    params = Hyperparams()
    
    # load preprocessed data
    train_processed, test_processed = load_data()
    X_train = train_processed[0]
    y_train = train_processed[1]
    
    X_test = test_processed[0]
    y_test = test_processed[1]

    # init CNN object
    model = cifar10_CNN_VGG_modified()
    # print model summary to console
    print(model.summary())
    # Data augmentation
    generator_train = ImageDataGenerator(
            rotation_range = 15,
            width_shift_range = 0.08,
            height_shift_range = 0.08,
            shear_range = 0.3,
            zoom_range = 0.08,
            horizontal_flip = True,
            vertical_flip = True,
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False
        ).flow(X_train, y_train, batch_size = params.batch_size)
    
    # if trained weights already exist
    if params.pretrain:
        # search phase 3, bc network crashed during second phase
        model.load_weights(params.inter_model[1])
        model.compile(
                loss=categorical_crossentropy,
                optimizer=rmsprop(lr=0.0003,decay=1e-6),
                metrics=['accuracy']
        )
        model.fit_generator(
            generator_train,
            steps_per_epoch=X_train.shape[0] // params.batch_size,
            epochs=params.epochs,
            validation_data=(X_test, y_test)
        )
        model.save_weights(params.inter_model[2])
    # train CNN from start
    else:
        ## Compile model with 3 search phases adapting learning rate for rmsprop algorithm
        
        # search phase 1
        model.compile(
            loss=categorical_crossentropy,
            optimizer=rmsprop(lr=0.001,decay=1e-6),
            metrics=['accuracy']
        )
        history1 = model.fit_generator(
            generator_train,
            steps_per_epoch=X_train.shape[0] // params.batch_size,
            epochs=2*params.epochs,
            validation_data=(X_test, y_test)
        )
        model.save_weights(params.inter_model[0])

        # search phase 2
        model.compile(
            loss=categorical_crossentropy,
            optimizer=rmsprop(lr=0.0006,decay=1e-6),
            metrics=['accuracy']
        )
        history2 = model.fit_generator(
            generator_train,
            steps_per_epoch=X_train.shape[0] // params.batch_size,
            epochs=params.epochs,
            validation_data=(X_test, y_test)
        )
        model.save_weights(params.inter_model[1])

        # search phase 3
        model.compile(
                loss=categorical_crossentropy,
                optimizer=rmsprop(lr=0.0003,decay=1e-6),
                metrics=['accuracy']
        )
        history3 = model.fit_generator(
            generator_train,
            steps_per_epoch=X_train.shape[0] // params.batch_size,
            epochs=params.epochs,
            validation_data=(X_test, y_test)
        )
        model.save_weights(params.inter_model[2])
        

    # model performance
    train_results = model.evaluate(X_train, y_train)
    test_results = model.evaluate(X_test, y_test)
    print("Train       Loss: ", train_results[0])
    print("Train       Acc: ",  train_results[1])
    print("Validation  Loss:",  test_results[0])
    print("Validation  Acc: ",  test_results[1])
    # saving archicteture:
    with open(params.architecture, 'w') as fp:
        json.dump(model.to_json(), fp)
    # saving histories
    if params.pretrain == False:
        with open(params.history[0], "w") as jp1:
            json.dump(history1.history, indent = 4, separators = (',', ': '), fp = jp1)
        with open(params.history[1], "w") as jp2:
            json.dump(history2.history, indent = 4, separators = (',', ': '), fp = jp2)
        with open(params.history[2], "w") as jp3:
            json.dump(history3.history, indent = 4, separators = (',', ': '), fp = jp3)        
    
    
    ## todo concatenate all histories
    
    ## plot
    if False:
        ## Training / Test wrt to epoch accuracy and loss plot:
        fig, axes = plt.subplots(figsize = (16, 8), ncols = 2)
        # plot accuracy for train and test set
        ax = axes[0]
        ax.plot(history.history["acc"])
        ax.plot(history.history["val_acc"])
        ax.set_title("accuracy")
        ax.set_ylabel("accuracy")
        ax.set_xlabel("epoch")
        ax.legend(["train', 'test"], loc = "lower right")
    
        # plot loss for train and test set
        ax = axes[1]
        ax.plot(history.history["loss"])
        ax.plot(history.history["val_loss"])
        ax.set_title("loss")
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        ax.legend(["train", "test"], loc = "upper right")
        
        # save
        fig.savefig("../model/history.png")   
        plt.close(fig)
    

# run main program when calling main.py programm in shell
if __name__ == '__main__':
    main()

