import os
import gym
import cv2
import random
import argparse
import sys, glob
import numpy as np
import cPickle as pickle
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Convolution2D
import memory

#Script Parameters
input_dim = 80 * 80
gamma = 0.99
update_frequency = 1
resume = False
render = False
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
EXPLORE = 100000
explorationRate = INITIAL_EPSILON
stepCounter = 0
current_epoch = 0
loadsim_seconds = 0
minibatch_size = 1

class DQN:

    
    def __init__(self, n_output, memorySize=100000, discountFactor=0.95, learningRate=1e-3, learnStart=1000):
        """
        Parameters:
        - outputs : output size
        - discountFactor : the discount factor (gamma)
        - learningRate : learning rate
        - learnStart : steps to happen before for learning Set to 128
        """
        self.output_size = n_output
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.learnStart = learnStart
        self.explorationRate = INITIAL_EPSILON

    def __call__(self,state):
        qvalues = self.getQValues(state)
        return self.selectAction(qvalues, self.explorationRate)

    def initNetworks(self):
        model = self.createModel()
        self.model = model
    
    def createModel(self,input_dim=80*80, model_type=1):
        model = Sequential()
        if model_type==0:
            model.add(Reshape((1,80,80), input_shape=(input_dim,)))
            model.add(Flatten())
            model.add(Dense(200, activation = 'relu'))
            model.add(Dense(self.output_size, activation='softmax'))
            opt = RMSprop(lr=self.learningRate)
        else:
            model.add(Reshape((1,80,80), input_shape=(input_dim,)))
            '''
            model.add(Convolution2D(16, 3, 3, subsample=(2,2)))
            model.add(Activation('relu'))
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(16, 3, 3, subsample=(2,2)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
            model.add(Flatten())
            model.add(Dense(256))
            model.add(Activation('relu'))
            model.add(Dense(network_outputs))
            adam = Adam(lr=self.learningRate)
            model.compile(loss='mse',optimizer=adam)
            #model.compile(RMSprop(lr=self.learningRate), 'MSE')
            model.summary()
            '''
            model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
            model.add(Flatten())
            model.add(Dense(16, activation='relu', init='he_uniform'))
            model.add(Dense(self.output_size, activation='softmax'))
            opt = Adam(lr=self.learningRate)
            
            model.compile(loss='categorical_crossentropy', optimizer=opt)
            
        return model
    
    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ",i,": ",weights
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        #print state.shape
        predicted = self.model.predict(state.reshape([1,state.shape[0]]),batch_size=1)
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)
        return predicted[0]
        
    def getMaxQ(self, qValues):
        return np.max(qValues)
        
    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1
    def set_reward(self, reward):
        self.reward = reward

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # print(miniBatchSize)
        # Do not learn until we've got self.learnStart samples        
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            '''
            X_batch = np.empty((6400,), dtype = np.float64)
            Y_batch = np.empty((1,self.output_size), dtype = np.float64)
            '''
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)
                #print(state.copy().shape)
                '''
                X_batch = np.append(X_batch, state.copy(), axis=0)
                print(X_batch.shape)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                print(Y_batch.shape)
                '''
                X_batch =  state.copy()
                #print(X_batch.shape)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.array([Y_sample])
                #print(Y_batch.shape)
                '''
                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
                '''
                #X_batch = X_batch.reshape([1,X_batch[0]])
                #print len(miniBatch)
            self.model.fit(X_batch.reshape([1,X_batch.shape[0]]), Y_batch, validation_split=0.2, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())
    
    def reinforcement_train(self): 
        self.addMemory(self.state,self.action, self.reward, self.observation, self.done) 
        stepCounter = self.frame
        if self.explorationRate > FINAL_EPSILON and stepCounter > self.learnStart:
            self.explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        
        if stepCounter == self.learnStart:
            print("Starting learning")

        if stepCounter >= self.learnStart:
            self.learnOnMiniBatch(minibatch_size, False)
        
    def get_info(self, state, action, reward, observation, done, frame):
        self.state = state
        self.action = action
        self.reward = reward
        self.observation = observation
        self.done = done
        self.frame = frame

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)

    

