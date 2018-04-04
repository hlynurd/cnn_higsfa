import numpy as np
from sklearn.decomposition import PCA
import mdp
import glob
import os
import hashlib
import time
from scipy import misc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def get_kelly_colors():
# theory - https://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf (page 5)
# kelly's colors - https://i.kinja-img.com/gawker-media/image/upload/1015680494325093012.JPG
# hex values - http://hackerspace.kinja.com/iscc-nbs-number-hex-r-g-b-263-f2f3f4-242-243-244-267-22-1665795040

    kelly_colors = ['#222222', '#BE0032', '#008856', '#E68FAC', '#0067A5',  '#604E97', '#F6A600', '#DCD300', '#882D17', '#8DB600', '#654522', '#E25822', '2B3D26', '#F2F3F4', '#848482', '#C2B280', '#A1CAF1', '#F99379', '#875692', '#F38400', '#B3446C', '#F3C300']
    return kelly_colors

def prepare_folders_mnist(current_model, per_character_sample):
    data_root = 'experimental_data_mnist'
    make_dir_if_not_exists(data_root)
    experiment_root = data_root + "/" + current_model
    make_dir_if_not_exists(experiment_root)
    for samples_per_character in per_character_sample:
        samples_root = experiment_root + "/" + str(samples_per_character) + "_characters"
        make_dir_if_not_exists(samples_root)
        make_dir_if_not_exists(samples_root + "/" + "training")
        make_dir_if_not_exists(samples_root + "/" + "testing")
    return True

def make_dir_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def prepare_time():
    hash = hashlib.sha1()
    hash.update(str(time.time()))
    return hash.hexdigest()[:10]

def get_list_of_mnist_indices(Y, number_of_stacked_index_copies = 1):
    lst  = []
    for i in range(Y.shape[0]):
        lst.append(np.argmax(Y[i,:]))
    def all_indices(lst, a):
        result = []
        for i, x in enumerate(lst):
            if x==a:
                result.append(i)
        return result
    indices = [[] for x in xrange(10)]
    for i in range(10):
        for j in range(number_of_stacked_index_copies):
            indices[i] = all_indices(lst, i) #+ all_indices(lst, i)
    return indices

def get_hgsfa_features_and_normalize(training_data, testing_data, flow):
    train_hgsfa_features = flow.execute(training_data.reshape(training_data.shape[0], training_data.shape[1]*training_data.shape[2]))
    test_hgsfa_features = flow.execute(testing_data.reshape(testing_data.shape[0], testing_data.shape[1]*testing_data.shape[2]))
    train_hgsfa_features = train_hgsfa_features - np.mean(train_hgsfa_features, axis=0)
    train_hgsfa_features = train_hgsfa_features / np.std(train_hgsfa_features, axis=0)

    test_hgsfa_features = test_hgsfa_features - np.mean(test_hgsfa_features, axis=0)
    test_hgsfa_features = test_hgsfa_features / np.std(test_hgsfa_features, axis=0)
    return train_hgsfa_features, test_hgsfa_features


# not being used
def get_cycle_indices(Y):
    none_hot_Y  = []
    for i in range(Y.shape[0]):
        none_hot_Y.append(np.argmax(Y[i,:]))
    copied = list(none_hot_Y)
    permuted_indices = []
    dels = 0
    for i in range(len(copied)):
        tag_for_this_element = i%10
        try: 
            index = copied.index(tag_for_this_element)
            copied[index] = -1
            permuted_indices.append(index)
        except:
            continue
    p_n = len(permuted_indices)
    c_n = len(none_hot_Y)
    for i in range(c_n - p_n):
        permuted_indices.append(none_hot_Y[p_n + i])
    return permuted_indices

def get_train_test_mnist_split(X, XX, Y, YY, training_size, model_nr = 0):
    X = X.reshape(X.shape[0], 28, 28,1)
    XX = XX.reshape(XX.shape[0], 28, 28,1)
    none_hot_Y  = []
    none_hot_YY = []
    for i in range(Y.shape[0]):
        none_hot_Y.append(np.argmax(Y[i,:]))
    for i in range(YY.shape[0]):
        none_hot_YY.append(np.argmax(YY[i,:]))

    sorted_labels = sorted(none_hot_Y)
    sorted_indices = [b[0] for b in sorted(enumerate(none_hot_Y),key=lambda i:i[1])]
    sorted_Y = Y[sorted_indices,:]
    sorted_X = X[sorted_indices,:]

    sorted_labels = sorted(none_hot_YY)
    sorted_indices = [b[0] for b in sorted(enumerate(none_hot_YY),key=lambda i:i[1])]
    sorted_YY = YY[sorted_indices,:]
    sorted_XX = XX[sorted_indices,:]
    
    
    cumulative_Y = []
    cumulative_Y.append(none_hot_Y.count(0))
    cumulative_YY = []
    cumulative_YY.append(none_hot_YY.count(0))
    for i in range(1, 10):
        cumulative_Y.append(none_hot_Y.count(i) + cumulative_Y[i-1])
        cumulative_YY.append(none_hot_YY.count(i) + cumulative_YY[i-1])

    class_train_size = training_size / 10
    offset = np.remainder(model_nr * class_train_size, cumulative_Y[0])
    upper_limit = np.minimum(class_train_size+offset, cumulative_Y[0])
    
    train_indices = range(0+offset, upper_limit)
    
    #case for wrapping around
    print(upper_limit)
    print(model_nr * class_train_size)
    print(cumulative_Y[0])
    if upper_limit >=cumulative_Y[0]:
        print("wrap")
        print(cumulative_Y[0]-(class_train_size+offset))
        #print(cumulative_Y[0] - upper_limit)
        #print(range(0, cumulative_Y[0] - upper_limit))
        train_indices = train_indices + range(0,  (class_train_size+offset) - cumulative_Y[0])
    
    for i in range(1, 10):
        
        offset = np.remainder(model_nr * class_train_size, cumulative_Y[i])
        upper_limit = np.minimum(cumulative_Y[i-1]+class_train_size + offset , cumulative_Y[i])
        
        print(i)
        print(cumulative_Y[i-1] + offset)
        print(upper_limit)
        print(model_nr * class_train_size)
        print(cumulative_Y[i])
        print(cumulative_Y[i]-(class_train_size+offset))
        print(range(cumulative_Y[i-1] + offset,
                        upper_limit))
        print("\n")
        
        train_indices = train_indices + range(cumulative_Y[i-1] + offset,
                        upper_limit)
        
        if upper_limit >=cumulative_Y[i]:
            print("wrap")
            train_indices = train_indices + range(cumulative_Y[i-1], (class_train_size+offset) - cumulative_Y[i])
            
        #case for wrapping around
        
    print(train_indices)
        
    if training_size != 60000:
        train_x = sorted_X[train_indices,:]
        train_y = sorted_Y[train_indices,:]
    else:
        train_x = sorted_X[:,:]
        train_y = sorted_Y[:,:]
        
    test_x = sorted_XX[:,:]
    test_y = sorted_YY[:,:]
    
    none_hot_train_y  = []
    for i in range(np.array(train_y).shape[0]):
        none_hot_train_y.append(np.argmax(train_y[i,:]))
        
    return train_x, train_y, test_x, test_y, none_hot_train_y


def graph_delta_values(y, edge_weights):
    """ Computes delta values from an arbitrary graph as in the objective 
    function of GSFA. The feature vectors are not normalized to weighted 
    unit variance or weighted zero mean.
    """
    R = 0
    deltas = 0
    for (i, j) in edge_weights.keys():
        w_ij = edge_weights[(i, j)]
        deltas += w_ij * (y[j] - y[i]) ** 2
        R += w_ij
    return deltas / R

def PCA_then_polynomial_expand_data(data, pca_dims, polynomial_order, pca = None):
    if pca is None:
        pca = PCA(n_components=pca_dims)
        pca_data = pca.fit_transform(data)
    else:
        pca_data = pca.transform(data)
    expansion_node = mdp.nodes.PolynomialExpansionNode(polynomial_order)
    expanded_pca_data = expansion_node.execute(pca_data)
    return expanded_pca_data, pca



def to_one_hot(array_):
    n_values = np.max(array_) + 1
    one_hot_array = np.eye(n_values)[array_]
    return one_hot_array

def standardize(array_):
    std_array = array_ - np.mean(array_)
    std_array= std_array / np.std(std_array)
    return std_array 

def prepare_omniglot_task(tasknr):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    cnt = -1
    for filename in glob.iglob('Omniglot/all_runs/run'+tasknr+'/*'): 
        for filename1 in glob.iglob(filename+'/*'):
            cnt = cnt + 1 
            image = misc.imread(filename1)
            image = misc.imresize(image, (35, 35))
            item_name = filename1.split('/')[4]
            item_class = -1
            cnt = -1
            with open('Omniglot/all_runs/run'+tasknr+'/class_labels.txt', "r") as ins:
                for line in ins:
                    cnt = cnt + 1
                    if item_name in line:
                        item_class = cnt
            if filename.split('/')[3] == 'test':
                test_x.append(image)
                test_y.append(item_class)
            else:
                train_x.append(image)
                train_y.append(item_class)
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    return train_x, test_x, train_y, test_y

def write_mnist_results(current_model, sample_size, trial, data_part, data_root, training_predictions, train_y, training_indices = None):
    hits = 0
    
    append_write = 'w'
    filename = "run_" + str(trial)
    
    path = data_root  + current_model + "/" + str(sample_size) + "_characters/" + data_part + "/" + filename

    for i,row in enumerate(training_predictions):
        row_list = list(row)
        train_yi_list = list(train_y[i])
        true_label = train_yi_list.index(max(train_yi_list))
        predicted_label = row_list.index(max(row_list))
        if os.path.exists(path):
            append_write = 'a' # append if already exists
        highscore = open(path, append_write)
        if append_write == 'w':
            highscore.write("id, true label, predicted label \n")
        if data_part == "training" and training_indices != None:
            highscore.write("%i, %i, %i \n" % (training_indices[i], true_label, predicted_label))
        else:
            highscore.write("%i, %i, %i \n" % (i, true_label, predicted_label))
        highscore.close()
        if true_label == predicted_label:
            hits += 1
    return hits / float(len(training_predictions)) # return (hits / float(len(training_predictions)))
    


def get_cnn_features_and_normalize(training_data, testing_data, new_model):
    
    train_hgsfa_features = new_model.predict(np.reshape(training_data, ([training_data.shape[0], 35, 35, 1]))[:, :, :, :])
    train_hgsfa_features = standardize(train_hgsfa_features)
    
    test_hgsfa_features = new_model.predict(np.reshape(testing_data, ([testing_data.shape[0], 35, 35, 1]))[:, :, :, :])
    test_hgsfa_features = standardize(test_hgsfa_features)
    return train_hgsfa_features, test_hgsfa_features




def log_results(alphabets, per_alphabet_character, per_character_sample, classic_accuracies, trial0_accuracies, trial1_accuracies, trial2_accuracies, max_trials, filename):
    
        if os.path.exists(filename):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
            
        highscore = open(filename,append_write)
        print("---- Alphabets: %i, characters per alphabet: %i, samples per character: %i ----" % (alphabets, per_alphabet_character, per_character_sample))
        highscore.write("\n Class: %i, pc_samples: %i \n" % (alphabets, per_character_sample))
        
        print("-- Classic task--")                       #
        highscore.write("-- Classic Omniglot task-- \n")          #
        mean = np.mean(classic_accuracies)
        interval_offset = 1.96 * np.sqrt(mean * (1-mean) / float(max_trials * 20 * 9))
        mean_print     = "Mean       : %.4f" % mean
        print(mean_print)                                               #
        highscore.write(mean_print + "\n ")                        #
        std_print      = "Std. dev.  : %.4f" % np.std(classic_accuracies)
        print(std_print)                         #
        highscore.write(std_print + " \n")  #
        interval_print = "95 interval: %.4f" % interval_offset
        print(interval_print)                                    #
        highscore.write(interval_print + " \n")             #
        
        print("-- Task 0  --")
        highscore.write("-- Trial 0  -- \n")
        mean = np.mean(trial0_accuracies) 
        interval_offset = 1.96 * np.sqrt(mean * (1-mean) / float(max_trials * 9))
        mean_print     = "Mean       : %.4f" % mean
        print(mean_print)                                               #
        highscore.write(mean_print + "\n ")                        #
        std_print      = "Std. dev.  : %.4f" % np.std(trial0_accuracies)
        print(std_print)                         #
        highscore.write(std_print + " \n")  #
        interval_print = "95 interval: %.4f" % interval_offset
        print(interval_print)                                    #
        highscore.write(interval_print + " \n")             #
        
        print("-- Task 1  --")
        highscore.write("-- Trial 1  -- \n")
        mean = np.mean(trial1_accuracies) 
        interval_offset = 1.96 * np.sqrt(mean * (1-mean) / float(max_trials * 9))        
        mean_print     = "Mean       : %.4f" % mean
        print(mean_print)                                               #
        highscore.write(mean_print + "\n ")                        #
        std_print      = "Std. dev.  : %.4f" % np.std(trial1_accuracies)
        print(std_print)                         #
        highscore.write(std_print + " \n")  #
        interval_print = "95 interval: %.4f" % interval_offset
        print(interval_print)                                    #
        highscore.write(interval_print + " \n")             #
        
        print("-- Task 2  --")
        highscore.write("-- Trial 2  -- \n")
        mean = np.mean(trial2_accuracies) 
        interval_offset = 1.96 * np.sqrt(mean * (1-mean) / float(max_trials * 20 * 9))        
        mean_print     = "Mean       : %.4f" % mean
        print(mean_print)                                               #
        highscore.write(mean_print + "\n ")                        #
        std_print      = "Std. dev.  : %.4f" % np.std(trial2_accuracies)
        print(std_print)                         #
        highscore.write(std_print + " \n")  #
        interval_print = "95 interval: %.4f" % interval_offset
        print(interval_print)                                    #
        highscore.write(interval_print + " \n")             #
        
        print("----------------")
        highscore.write("-------- \n")
        

        
        
        
        
            
        highscore.close()


        
