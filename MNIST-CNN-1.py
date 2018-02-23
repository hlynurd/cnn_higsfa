
import matplotlib.pyplot as plt
import mdp
import numpy as np
import scipy
from keras.models import Sequential
from datetime import datetime
from keras.layers import *
from utils import *
import scipy.interpolate as interpolate
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from experimental_models import *
import random
from keras.callbacks import EarlyStopping
random.seed(0)

X = np.load('X.npy')
XX = np.load('XX.npy')
Y = np.load('Y.npy')
YY = np.load('YY.npy')
max_trials = 100
sample_sizes = [4000] # 50, 200, 500, 2000, 6000]
current_model = "cnn_1"
prepare_folders_mnist(current_model, sample_sizes)
#for sample_size in sample_sizes:
for trial in range(max_trials):
    train_accuracies = []
    test_accuracies = []
    
#    for trial in range(max_trials):
    for sample_size in sample_sizes:
        print("Samples per character: %i, trial: %i" % (sample_size, trial))
        if sample_size < 6000:
            indices = get_list_of_mnist_indices(Y, number_of_stacked_index_copies = 1)
            training_indices = []
            validation_indices = []
            for i in range(10):
                random.shuffle(indices[i])
                training_indices = training_indices + indices[i][:sample_size]
                validation_indices = validation_indices + indices[i][sample_size:sample_size+1000]
            train_x = X[training_indices, :, :, :]  #[:sample_sizes, :, :, :]
            train_y = Y[training_indices, :]        #[:sample_sizes, :]i
            val_x = X[validation_indices, :, :, :]
            val_y = Y[validation_indices, :]

            test_x   = XX
            test_y   = YY
        else:
            train_x = np.array(X)
            train_y = np.array(Y)
            val_x   = np.array(XX)
            val_y   = np.array(YY)


        net = instantiate_cnn1(1, 10, 28, 28)
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        #net.fit(np.reshape(train_x, ([train_x.shape[0], 28, 28, 1])), train_y, epochs=20,
        #        batch_size=8, verbose = 0)
#        net.fit(np.reshape(train_x, ([train_x.shape[0], 28, 28, 1])), train_y, validation_data=(val_x, val_y),
        if sample_size < 6000:
            net.fit(np.reshape(train_x, ([train_x.shape[0], 28, 28, 1])), train_y, validation_data=(val_x, val_y), callbacks=[early_stopping], batch_size=128, verbose = 1, epochs = 1000, shuffle=True)
        else:
            net.fit(np.reshape(train_x, ([train_x.shape[0], 28, 28, 1])), train_y, validation_split=1.0/6.0, callbacks=[early_stopping], batch_size=16, verbose = 1, epochs = 1000, shuffle=True)




        #eval_1 = net.evaluate(np.reshape(train_x, ([train_x.shape[0], 28, 28, 1])), train_y, verbose=0)
        training_predictions = list(net.predict(train_x))

        data_root = "experimental_data_mnist/"
        if sample_size < 6000:
            accuracy = write_mnist_results(current_model, sample_size, trial,
                                           "training", data_root, training_predictions, train_y, training_indices)
        else:
            accuracy = write_mnist_results(current_model, sample_size, trial,
                                           "training", data_root, training_predictions, train_y)

        
        train_accuracies.append(accuracy)
        #eval_2 = net.evaluate(np.reshape(val_x, ([val_x.shape[0], 28, 28, 1])), val_y, verbose=0)
        testing_predictions = net.predict(test_x)
        accuracy = write_mnist_results(current_model, sample_size, trial,
                                       "testing", data_root, testing_predictions, test_y)
        test_accuracies.append(accuracy)
        
        print(datetime.now())
        print("training accuracies per trial: " + str(train_accuracies))
        print("test accuracies per trial:     " + str(test_accuracies))
        print("\n")
