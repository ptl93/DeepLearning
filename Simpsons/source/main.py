# -*- coding: utf-8 -*-
"""
Main python programm for classifying Simpsons dataset

@author: Tuan Le
@email: tuanle@hotmail.de
"""
from loader import load_data
from model import simpsons_CNN

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
        # training params
        self.batch_size = 128
        self.epochs = 50
        self.input_shape = (64, 64, 3)
        self.classes = 18

# JSON file saver
import json
def writeToJSONFile(path, fileName, data):
    filePathNameWExt = "./" + path + "/" + fileName + ".json"
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)
        
        
# python main programm function
def main():
    # init Hyperparams object
    params = Hyperparams()
    
    # load preprocessed data
    (X_train, y_train), (X_test, y_test) = load_data()
    
    # init CNN object
    model = simpsons_CNN()
    # print model summary to console
    print(model.summary())
    # if trained weights already exist
    if params.pretrain:
        model.load_weights(params.inter_model)
        model.compile(
            loss = categorical_crossentropy,
            optimizer = adam(lr = 1e-3, decay = 1e-4),
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
            optimizer = adam(lr = 1e-3, decay = 1e-4),
            metrics = ["accuracy"]
        )
        history = model.fit_generator(
            generator_train,
            X_train.shape[0] // params.batch_size,
            epochs = params.epochs,
            validation_data = (X_test, y_test),
            validation_steps = (X_test.shape[0] // params.epochs),
            callbacks = callbacks
        )

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
    # saving history
    if params.pretrain == False:
        history_json = json.dumps(history.history, indent = 4, separators=(',', ': '))
        writeToJSONFile("./model", "evalHistory", history_json)
    
# run main program when calling main.py programm in shell
if __name__ == '__main__':
    main()

