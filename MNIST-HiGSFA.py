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
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

import random
random.seed(0)

X = np.load('X.npy')
XX = np.load('XX.npy')
Y = np.load('Y.npy')
YY = np.load('YY.npy')
max_trials = 100
sample_sizes = [4000] #[3, 10, 50, 200, 500, 2000, 6000] # 50, 200, 500, 2000, 6000]
current_model = "higsfa"
prepare_folders_mnist(current_model, sample_sizes)
for trial in range(0, max_trials):
    train_accuracies = []
    test_accuracies = []
    for sample_size in sample_sizes:
        print("Samples per character: %i, trial: %i" % (sample_size, trial))
        test_x = XX
        test_y = YY
        if sample_size < 6000:
            indices = get_list_of_mnist_indices(Y, number_of_stacked_index_copies = 1)
            training_indices = []
            validation_indices = []
            for i in range(10):
                random.shuffle(indices[i])
                training_indices = training_indices + indices[i][:sample_size]
                validation_indices = validation_indices + indices[i][sample_size:sample_size+1000]
            train_x = X[training_indices, :, :, :]  #[:sample_sizes, :, :, :]
            train_y = Y[training_indices, :]        #[:sample_sizes, :]
            
            val_x = X[validation_indices, :, :, :]  #[:sample_sizes, :, :, :]
            val_y = Y[validation_indices, :]        #[:sample_sizes, :]
            
        else:
            train_x = X
            train_y = Y
            indices = get_list_of_mnist_indices(Y, number_of_stacked_index_copies = 1)
            validation_indices = []
            for i in range(10):
                random.shuffle(indices[i])
                validation_indices = validation_indices + indices[i][0:0+1000]
            val_x = X[validation_indices, :, :, :]
            val_y = Y[validation_indices, :]

        flow = instantiate_higsfa(28, 28)
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1]*test_x.shape[2])
        val_x = val_x.reshape(val_x.shape[0], val_x.shape[1]*val_x.shape[2])
        none_hot_Y  = []
        none_hot_YY = []
        none_hot_val_y = []
        for i in range(train_y.shape[0]):
            none_hot_Y.append(np.argmax(train_y[i,:]))
        for i in range(val_y.shape[0]):
            none_hot_val_y.append(np.argmax(val_y[i,:]))
        for i in range(YY.shape[0]):
            none_hot_YY.append(np.argmax(YY[i,:]))    
            
        block_sizes_train = []    
        for i in range(10):
            block_sizes_train.append(none_hot_Y.count(i))
        
        params_set = {"train_mode":"clustered", "block_size" : block_sizes_train}
        try:
            flow.special_train_cache_scheduler_sets(train_x, params_set)
        except:
            print("oh boy an svd error")
            continue
        try:
            train_hgsfa_features = flow.execute(train_x)
            train_hgsfa_features = standardize(train_hgsfa_features)
        except:
            print("ok1")
            continue        
        test_hgsfa_features = flow.execute(test_x)
        test_hgsfa_features = standardize(test_hgsfa_features)
        val_hgsfa_features = flow.execute(val_x)
        val_hgsfa_features = standardize(val_hgsfa_features)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print more_nodes.compute_flow_size(flow)
        
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        net = Sequential()        
        net.add(Dense(10, input_dim=train_hgsfa_features.shape[1], activation='softmax'))
        net.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print("training with %i samples" % sample_size)
         
        net.fit(np.reshape(train_hgsfa_features, ([train_hgsfa_features.shape[0], train_hgsfa_features.shape[1]])), train_y, validation_data=(np.reshape(val_hgsfa_features, ([val_hgsfa_features.shape[0], val_hgsfa_features.shape[1]])), val_y), callbacks=[early_stopping], batch_size=128, verbose = 1, epochs = 1000, shuffle=True)

        training_predictions = list(net.predict(np.reshape(train_hgsfa_features, ([train_hgsfa_features.shape[0],
                                                            train_hgsfa_features.shape[1]]))))
        data_root = "experimental_data_mnist/"
        if sample_size < 6000:
            accuracy = write_mnist_results(current_model, sample_size, trial,
                                           "training", data_root, training_predictions, train_y, training_indices)
        else:
            accuracy = write_mnist_results(current_model, sample_size, trial,
                                           "training", data_root, training_predictions, train_y)

#        accuracy = write_mnist_results(current_model, sample_size, trial,
#                                       "training", data_root, training_predictions, train_y, training_indices)
        
        train_accuracies.append(accuracy)

        testing_predictions = net.predict(np.reshape(test_hgsfa_features,
                 ([test_hgsfa_features.shape[0], test_hgsfa_features.shape[1]])))
        accuracy = write_mnist_results(current_model, sample_size, trial,
                                       "testing", data_root, testing_predictions, test_y)
        test_accuracies.append(accuracy)
        
    print(datetime.now())
    print("training accuracies per trial: " + str(train_accuracies))
    print("test accuracies per trial:     " + str(test_accuracies))
    print("\n")
