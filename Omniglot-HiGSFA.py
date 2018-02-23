import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from experimental_models import *
from utils import *
from utils_omniglot import *
alphabet_numbers        = [4, 6, 8, 10, 12] #[4, 6, 8, 10, 12] 
per_alphabet_characters = [4, 6, 8, 10, 12] #[4, 6, 8, 10, 12]
per_character_samples   = [4, 16] # [4, 16] 
max_trials              = 100
seed                    = 0
n_way                   = 16
current_model           = "higsfa"
challenges              = ["challenge0", "challenge1", "challenge2"]
prepare_folders(current_model, alphabet_numbers, per_alphabet_characters, per_character_samples, challenges)

for alphabets in alphabet_numbers:
    for characters_per_alphabet in per_alphabet_characters:
        for per_character_sample in per_character_samples:
            
            if not characters_per_alphabet == 8 and not alphabets == 8:
                continue
            print("---- Alphabets: %i, characters per alphabet: %i, samples per character: %i ----" % 
                  (alphabets, characters_per_alphabet, per_character_sample))
            accuracies_cnn_c0 = []
            accuracies_cnn_c1 = []
            accuracies_cnn_c2 = []
            for trial in range(max_trials):
                hashtime = prepare_time()
                seed = seed + 1
                test_x = np.array([])
                test_y = np.array([])

                training_root = 'Omniglot/images_background/*'
                alphabet_paths, char_paths = get_random_paths(training_root, alphabets,
                                                              characters_per_alphabet, seed)                
                train_x, train_y, sample_lists = prepare_training_data(char_paths, per_character_sample, seed)

                
                one_hot_y = to_one_hot(train_y)
                #train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
                #train_x = standardize(train_x)
                #test_x = standardize(test_x)

                train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
                block_sizes_train = [per_character_sample] * (train_x.shape[0] / per_character_sample)

                flow = instantiate_higsfa()
                params_set = {"train_mode":"clustered", "block_size" : block_sizes_train}
                flow.special_train_cache_scheduler_sets(train_x/255.0, params_set)

                del train_x; del train_y

                # -------- CHALLENGE 0 -----------
                challenge_step = 0
                challenge_number = 0
                for challenge_step in range(per_character_sample/2):
                    
                    accuracies_cnn_c0 = do_challenge(current_model, challenge_number, sample_lists, alphabets,
                                                     characters_per_alphabet, per_character_sample,
                                                     challenge_step, n_way, seed, None, hashtime,
                                                     accuracies_cnn_c0, False, flow)
  
                    
                # -------- CHALLENGE 1 -----------
                challenge_step = 0
                challenge_number = 1
                for challenge_step in range((20-per_character_sample)/2):
                    accuracies_cnn_c1 = do_challenge(current_model, challenge_number, sample_lists, alphabets,
                                                     characters_per_alphabet, per_character_sample,
                                                     challenge_step, n_way, seed, None, hashtime,
                                                     accuracies_cnn_c1, False, flow)
                    

                # -------- CHALLENGE 2 -----------
                challenge_step = 0
                challenge_number = 2
                for challenge_step in range(5):
                    accuracies_cnn_c2 = do_challenge(current_model, challenge_number, sample_lists, alphabets,
                                                     characters_per_alphabet, per_character_sample,
                                                     challenge_step, n_way, seed, None, hashtime,
                                                     accuracies_cnn_c2, False, flow)

            log_results(alphabets, characters_per_alphabet, per_character_sample, [], accuracies_cnn_c0,
                        accuracies_cnn_c1, accuracies_cnn_c2, max_trials, current_model + '_results.txt')
            print(str(datetime.now()))
