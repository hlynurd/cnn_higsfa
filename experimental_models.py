import mdp
from cuicuilco import system_parameters
from cuicuilco import network_builder
from cuicuilco import sfa_libs
from cuicuilco import more_nodes
from cuicuilco import patch_mdp
import keras
from cuicuilco.nonlinear_expansion import *
from keras import Sequential
from keras.layers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))



def instantiate_cnn1(alphabets, per_alphabet_character, dim1=35, dim2=35):
#    import tensorflow as tf
    tf.reset_default_graph()
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    config.log_device_placement=True

#    sess = tf.Session(config=config)  #With the two options defined above
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

#    with tf.Session() as sess:
    net = Sequential()
    net.add(Conv2D(16, (7, 7), input_shape=(dim1, dim2, 1), activation='relu', padding= "same"))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Conv2D(32, (5, 5), activation='relu', padding = "same"))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Conv2D(32, (5, 5), activation='relu', padding = "same"))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Flatten())
    net.add(Dense(150, activation='relu'))
    net.add(Dense(alphabets*per_alphabet_character, activation='softmax'))
    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return net


def instantiate_cnn2(alphabets, per_alphabet_character, dim1=35, dim2=35):
#    import tensorflow as tf
    tf.reset_default_graph()
 #   config = tf.ConfigProto()
 #   config.gpu_options.allow_growth = True
 #   config.log_device_placement=True

#    sess = tf.Session(config=config)  #With the two options defined above
#
#    with tf.Session() as sess:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    net = Sequential()
    net.add(Conv2D(8, (7, 7), input_shape=(dim1, dim2, 1), activation='relu', padding = "same"))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Conv2D(16, (5, 5), activation='relu', padding = "same"))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Conv2D(16, (5, 5), activation='relu', padding = "same"))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Flatten())
    net.add(Dense(alphabets*per_alphabet_character, activation='softmax'))
    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])            
    return net


def instantiate_higsfa(dim1=35, dim2=35):
    maximum_delta = 1.99
    #expansion_function = swish #expansion_function = unsigned_08expo
    expansion_function = unsigned_08expo
    middle_features = 16
    ##################### First layer    #############################
    pSFALayerL0 = system_parameters.ParamsSFALayer()
    pSFALayerL0.x_field_channels=5
    pSFALayerL0.y_field_channels=5
    pSFALayerL0.x_field_spacing=2
    pSFALayerL0.y_field_spacing=2 
    pSFALayerL0.sfa_node_class = mdp.nodes.iGSFANode
    pSFALayerL0.sfa_args = {
        "slow_feature_scaling_method":None,
        "delta_threshold": maximum_delta,
        "reconstruct_with_sfa" : False}
    pSFALayerL0.sfa_out_dim = 25
    pSFALayerL0.clone_layer = True

    ##################### Second layer    ###########################

    pSFALayerL1 = system_parameters.ParamsSFALayer()
    pSFALayerL1.x_field_channels = 4
    pSFALayerL1.y_field_channels = 4
    pSFALayerL1.x_field_spacing= 2
    pSFALayerL1.y_field_spacing= 2
    pSFALayerL1.exp_funcs = [identity, expansion_function]
    #pSFALayerL1.sfa_node_class = mdp.nodes.GSFANode
    pSFALayerL1.sfa_node_class = mdp.nodes.iGSFANode
    pSFALayerL1.sfa_args = {
        "slow_feature_scaling_method":None,
        "delta_threshold": maximum_delta, 
        "reconstruct_with_sfa" : False,
        "max_length_slow_part":middle_features}
    pSFALayerL1.sfa_out_dim = middle_features
    pSFALayerL1.clone_layer = True


    Omnitglot_network = system_parameters.ParamsNetwork()
    Omnitglot_network.name = "Omniglot HiGSFA network"
    Omnitglot_network.layers = [pSFALayerL0,pSFALayerL1]
    flow, _, _ = network_builder.create_network(Omnitglot_network, dim1, dim2, None)
    return flow
