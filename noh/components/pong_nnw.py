import os
import gym
import cv2
import argparse
import sys, glob
import numpy as np
import cPickle as pickle
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Convolution2D

#Script Parameters
input_dim = 80 * 80
gamma = 0.99
update_frequency = 1
learning_rate = 0.001
resume = False
render = False

number_of_inputs = 6 #This is incorrect for Pong (?)
#number_of_inputs = 1


class NNW(object):
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        model = self.learning_model()
        self.model = model


    #Define the main model (WIP)
    def learning_model(self,input_dim=80*80, model_type=1):
        model = Sequential()
        if model_type==0:
            model.add(Reshape((1,80,80), input_shape=(input_dim,)))
            model.add(Flatten())
            model.add(Dense(200, activation = 'relu'))
            model.add(Dense(number_of_inputs, activation='softmax'))
            opt = RMSprop(lr=learning_rate)
        else:
            model.add(Reshape((1,80,80), input_shape=(input_dim,)))
            model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
            model.add(Flatten())
            model.add(Dense(16, activation='relu', init='he_uniform'))
            model.add(Dense(number_of_inputs, activation='softmax'))
            opt = Adam(lr=learning_rate)
            model.compile(loss='categorical_crossentropy', optimizer=opt)
            model.load_weights('pong_model_checkpoint.h5')
        return model
            
                                                                                        
    def __call__(self, state, **kwargs):
        aprob = ((self.model.predict(state.reshape([1,state.shape[0]]), batch_size=1).flatten()))
        action = np.random.choice(number_of_inputs, 1, p=aprob)[0]
        return action

    def set_reward(self, reward): pass

    def set_state(self, state): pass
    
    def supervised_train(self, data=None, label=None, epochs=None, **kwargs): pass
    def unsupervised_train(self, data=None, label=None, epochs=None, **kwargs): pass
    def reinforcement_train(self, data=None, label=None, epochs=None, **kwargs):  pass

    def reset(self): pass
