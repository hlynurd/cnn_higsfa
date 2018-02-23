import numpy as np
from sklearn.decomposition import PCA
import mdp
import glob
import os
import hashlib
import time
from scipy import misc
from sklearn.neighbors import NearestNeighbors
from utils import *
import matplotlib.pyplot as plt

def make_dir_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def prepare_time():
    hash = hashlib.sha1()
    hash.update(str(time.time()))
    return hash.hexdigest()[:10]

def prepare_folders(current_model, alphabet_numbers, per_alphabet_characters, per_character_samples, challenges):
    data_root = 'experimental_data'
    make_dir_if_not_exists(data_root)
    experiment_root = data_root + "/" + current_model
    make_dir_if_not_exists(experiment_root)
    for challenge in challenges:
        challenge_root = experiment_root + "/" + challenge
        make_dir_if_not_exists(challenge_root)
        
        for alphabets in alphabet_numbers:
            alphabet_number_root = challenge_root + "/" + str(alphabets) + "_alphabets"
            make_dir_if_not_exists(alphabet_number_root)
            
            for characters_per_alphabet in per_alphabet_characters:
                if not characters_per_alphabet == 8 and not alphabets == 8:
                    continue
                characters_number_root = alphabet_number_root + "/" + str(characters_per_alphabet) + "_characters"
                make_dir_if_not_exists(characters_number_root)
                
                for samples in per_character_samples:
                    samples_root = characters_number_root + "/" + str(samples) + "_samples"
                    make_dir_if_not_exists(samples_root)
                
    return True

def get_random_paths(root, classes, per_alphabet_characters, seed):
    np.random.seed(seed)
    character_roots = []
    alphabet_roots = []
    cnt = 0 
    character_count = 0
    
    traintrain_indices = range(classes * per_alphabet_characters)
    np.random.shuffle(traintrain_indices)
    traintrain_indices = traintrain_indices[:9]
    
    alphabets = []
    for filename in glob.iglob(root):    
        alphabets.append(filename)
    alphabets = np.array(alphabets)
    np.random.shuffle(alphabets)    
    
    for filename in alphabets:  
        
        if cnt == classes:
            break
        alphabet_roots.append(filename)    
        characters = []
        for filename1 in glob.iglob(filename+'/*'):    
            characters.append(filename1)
            
        characters = np.array(characters)
        np.random.shuffle(characters)   

        for filename1 in characters:
            character_roots.append(filename1)
            character_count = character_count + 1
            if character_count is per_alphabet_characters:
                character_count = 0
                break # this parts makes the "sub subsets" 
        cnt = cnt + 1
        
    return alphabet_roots, character_roots
    
def prepare_training_data(character_roots, per_character_samples, seed):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    sample_lists = []
    
    none_hot_vector_count  = -1    
    for path in character_roots:
        none_hot_vector_count += 1
        samples = []
        for filename in glob.iglob(path+'/*'):    
            samples.append(filename)
        samples = np.array(samples)
        np.random.shuffle(samples)     
        sample_lists.append(samples)
        
        samples_per_char_count = -1
        for datapoint in samples:
            samples_per_char_count += 1
            if samples_per_char_count == per_character_samples:
                break
            image = misc.imread(datapoint)
            image = misc.imresize(image, (35, 35))
            train_x.append(image)
            train_y.append(none_hot_vector_count)
    return np.array(train_x), np.array(train_y), sample_lists

def do_challenge(current_model, challenge_number, sample_lists, alphabets, characters_per_alphabet, per_character_sample, challenge_step, n_way, seed, new_model2, hashtime, accuracies, is_cnn = True, flow = None):
    challenges              = ["challenge0", "challenge1", "challenge2"]    
    if challenge_number == 0:
        c0_train_x, c0_train_y, c0_test_x, c0_test_y, shuffled_sample_lists = prepare_challenge0(sample_lists, per_character_sample, challenge_step, n_way, seed)
    elif challenge_number == 1:
        c0_train_x, c0_train_y, c0_test_x, c0_test_y, shuffled_sample_lists = prepare_challenge1(sample_lists, per_character_sample, challenge_step, n_way, seed)
    else:
        challenge2_root = 'Omniglot/images_evaluation/*'
        alphabet_paths, char_paths = get_random_paths(challenge2_root, alphabets, characters_per_alphabet, seed)                
        _, _, challenge2_lists = prepare_training_data(char_paths, per_character_sample, seed)
        c0_train_x, c0_train_y, c0_test_x, c0_test_y, shuffled_sample_lists = prepare_challenge2(
                                            challenge2_lists, per_character_sample, challenge_step, n_way, seed)
        
    if is_cnn:
        train_features, test_features = get_cnn_features_and_normalize(
            np.array(c0_train_x), np.array(c0_test_x), new_model2)
    else:
        train_features, test_features = get_hgsfa_features_and_normalize(
                                                                np.array(c0_train_x), np.array(c0_test_x), flow)
    actuals, closests = new_knn(train_features, test_features, 
                                c0_train_y, c0_test_y, shuffled_sample_lists)
    write_knn_results(hashtime, actuals, closests, current_model, challenges[challenge_number], alphabets,
                      characters_per_alphabet, per_character_sample)

    accuracies.append(do_knn_trial(train_features, test_features, 
                                          c0_train_y, c0_test_y))
    return accuracies
    


def prepare_challenge0(sample_lists, per_character_samples, step, n_way, seed):
    assert step < per_character_samples/2
    
    c0_train_x = []
    c0_train_y = []
    c0_test_x  = []
    c0_test_y  = []
    
    numpy_sample_lists = np.array(sample_lists)
    np.random.shuffle(numpy_sample_lists)
    
    none_hot_vector_count = 0
    
    for i in range(n_way):

        image = misc.imread(numpy_sample_lists[i, step])
        image = misc.imresize(image, (35, 35))
        c0_train_x.append(image)
        c0_train_y.append(none_hot_vector_count)
        
        image = misc.imread(numpy_sample_lists[i, step+1])
        image = misc.imresize(image, (35, 35))
        c0_test_x.append(image)
        c0_test_y.append(none_hot_vector_count)
        none_hot_vector_count += 1
    return np.array(c0_train_x), c0_train_y, np.array(c0_test_x), c0_test_y, numpy_sample_lists
    
    
def prepare_challenge1(sample_lists, per_character_samples, step, n_way, seed):    
    assert step < (20-per_character_samples) / 2
    
    c1_train_x = []
    c1_train_y = []
    c1_test_x  = []
    c1_test_y  = []
    
    numpy_sample_lists = np.array(sample_lists)
    np.random.shuffle(numpy_sample_lists)
    
    none_hot_vector_count = 0
    
    for i in range(n_way):

        image = misc.imread(numpy_sample_lists[i, step+per_character_samples])
        image = misc.imresize(image, (35, 35))
        c1_train_x.append(image)
        c1_train_y.append(none_hot_vector_count)
        
        image = misc.imread(numpy_sample_lists[i, step+per_character_samples+1])
        image = misc.imresize(image, (35, 35))
        c1_test_x.append(image)
        c1_test_y.append(none_hot_vector_count)
        none_hot_vector_count += 1
    return np.array(c1_train_x), c1_train_y, np.array(c1_test_x), c1_test_y, numpy_sample_lists

def prepare_challenge2(sample_lists, per_character_samples, step, n_way, seed):    
    c2_train_x = []
    c2_train_y = []
    c2_test_x  = []
    c2_test_y  = []
    
    numpy_sample_lists = np.array(sample_lists)
    np.random.shuffle(numpy_sample_lists)
    
    none_hot_vector_count = 0
    
    for i in range(n_way):

        image = misc.imread(numpy_sample_lists[i, step])
        image = misc.imresize(image, (35, 35))
        c2_train_x.append(image)
        c2_train_y.append(none_hot_vector_count)
        
        image = misc.imread(numpy_sample_lists[i, step+1])
        image = misc.imresize(image, (35, 35))
        c2_test_x.append(image)
        c2_test_y.append(none_hot_vector_count)
        none_hot_vector_count += 1
    return np.array(c2_train_x), c2_train_y, np.array(c2_test_x), c2_test_y, numpy_sample_lists

def new_knn(train_set, test_set, train_labels, test_labels, shuffled_sample_lists):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(train_set)
    distances, indices = nbrs.kneighbors(test_set)
    cnt = 0
    hit = 0
    actuals = []
    closests = []
    for tuple_ in indices:
        full_path = shuffled_sample_lists[int(test_labels[cnt]), 0]
        full_split = full_path.split("/")
        actual = full_split[2] + "/" + full_split[3]
        actuals.append(actual)
        
        full_path = shuffled_sample_lists[int(train_labels[tuple_[0]]), 0]
        full_split = full_path.split("/")
        closest = full_split[2] + "/" + full_split[3]
        closests.append(closest)
        cnt = cnt + 1
    return actuals, closests

def write_knn_results(hashtime, actuals, closests, model, challenge, a, cpa, s):
    for i in range(len(actuals)):
        append_write = 'w'
        path = "experimental_data/" + model + "/" + challenge + "/" + str(a) + "_alphabets/" + str(cpa) + "_characters/" + str(s) + "_samples/" + str(hashtime)
        if os.path.exists(path):
            append_write = 'a' # append if already exists

        highscore = open(path, append_write)
        highscore.write("%s, %s \n" % (actuals[i], closests[i]))
        highscore.close()
    return True


def do_knn_trial(train_set, test_set, train_labels, test_labels):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(train_set)
    distances, indices = nbrs.kneighbors(test_set)
    cnt = 0
    hit = 0
    for tuple_ in indices:
        if int(test_labels[cnt]) == int(train_labels[tuple_[0]]):
            hit = hit + 1
        cnt = cnt + 1
    return float(hit) / len(indices)

def visualize_task(trial2_train_x, trial2_test_x):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    plt.figure(1)
    for i in range(9):
        plt.subplot("33" + str(i))
        plt.axis('off')
        plt.imshow(trial2_train_x[i, :, :], cmap="gray")
        plt.show()
        print("\n")
        plt.figure(1)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for i in range(9):
        plt.subplot("33" + str(i))
        plt.axis('off')
        plt.imshow(trial2_test_x[i, :, :], cmap="gray")
        plt.show()
    print("-----------------------------------------------------")