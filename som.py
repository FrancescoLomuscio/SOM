# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from math import sqrt

import numpy as np
from numpy.linalg import norm
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
from multiprocessing import cpu_count, Process, Queue
import matplotlib.pyplot as plt

from datetime import datetime


from sklearn.datasets import load_iris
from sklearn.datasets import load_digits



def _build_iteration_indexes(data_len, num_iterations, random_generator=None):
    iterations = np.arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    return iterations

def euclidean_distance(x, w):
    return norm(np.subtract(x,w), axis = -1)

def cosine_distance(x, w):
    n = (x*w).sum(axis = 2)
    d = np.multiply(norm(x), norm(w,axis=2))
    return n / d

def manhattan_distance(x, w):
    return norm(np.subtract(x,w), ord=1, axis = -1)

class SOM:
    def __init__(self,x,y,input_dim,learning_rate,sigma,max_iter,distance_function):
        
        self._random_generator = np.random.RandomState(42)
        #Initialize class variables
        self._x = x
        self._y = y
        self._input_dim = input_dim
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._max_iter = max_iter
        self._distance_function = distance_function
        
        self._history = []
        
        self._neigx = np.arange(x)
        self._neigy = np.arange(y) 
        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
                
        #Create weights vector
        #Try to change range of weights in [-1,1] and normalize
        self._weights = np.random.rand(self._x,self._y,self._input_dim)
        #self._weights = self._weights*2-1
        #self._weights /= norm(self._weights, axis = -1, keepdims = True)
        
        self._activation_map = np.zeros((x, y))
        
    def decay(self,to_decay,n,max_iter):
        return to_decay*np.exp(-1*n/max_iter)
    
    def get_weights(self):
        return self._weights
    
    def activate(self, x):
        self._activation_map = self._distance_function(x, self._weights)
    
    def choose_winner(self,x):
        self.activate(x)
        return np.unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)
        
    def gaussian(self,c, sigma):
        d = 2*np.pi*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T
        
    def update(self,x,win,t,max_iter):
        lr = self.decay(self._learning_rate,t,max_iter)
        sigma = self.decay(self._sigma,t,max_iter)
        
        g = self.gaussian(win, sigma)*lr
        
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        return self._weights
            
    def train(self, data):
        iterations = _build_iteration_indexes(len(data), self._max_iter, 
                                              self._random_generator)
        #print(iterations)
        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.choose_winner(data[iteration]),
                                t, self._max_iter)
            self._history.append(self.quantization_error(data))
        print("Training FINITO\n")
            
    def plot(self,data,targets):
        colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan']
        markers = ['o'] * 10
        
        wm = self.winner_map(data)
        fig, ax = plt.subplots(figsize=(self._x,self._y))
        plt.pcolormesh(wm, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self._x + .5), range(self._x))
        plt.yticks(np.arange(.5, self._y + .5), range(self._y))
        #ax.set_aspect('equal')

        #fig, ax = plt.subplots(figsize=(5,5))
        for cnt, xx in enumerate(data):
            c = colors[targets[cnt]]
            w = self.choose_winner(xx)
            ax.plot(w[1] + .5 + 0.1 * np.random.randn(1), w[0] + .5 + 0.1 * np.random.randn(1),
                    markers[targets[cnt]], color=c, markersize=12)
        #plt.savefig("Map")
        plt.show()
        
        plt.plot(self._history)
        #plt.savefig("History")
        
                
    def winner_map(self, data):
        wm = np.zeros((self._x,self._y), dtype=int)
        for d in data:
            [x, y] = self.choose_winner(d)
            wm[x, y] += 1
        return wm
  
    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        #self._check_input_len(data)
        return norm(data-self.quantization(data), axis=1).mean()

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        #self._check_input_len(data)
        winners_coords = np.argmin(self._distance_from_weights(data), axis=1)
        return self._weights[np.unravel_index(winners_coords,
                                           self._weights.shape[:2])]       

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = np.array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        #print(np.shape(weights_flat))
        input_data_sq = np.power(input_data, 2).sum(axis=1, keepdims=True)
        #print(np.shape(input_data_sq))
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        #print(np.shape(weights_flat_sq))
        cross_term = np.dot(input_data, weights_flat.T)
        #print(np.shape(cross_term))
        return np.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
    
def main():
    dataset = load_iris()
    #dataset = load_digits()

    x = dataset.data[:, :]
    print("EUCLIDEAN DISTANCE")
    start_1=datetime.now()
    som = SOM(5,5,np.shape(x)[1],0.1,0.8,5000,euclidean_distance)
    y = dataset.target
    som.train(x)
    som.plot(x,y)
    time_1 = datetime.now()-start_1
    print(time_1)
    
    print("COSINE DISTANCE")
    start_2=datetime.now()
    som = SOM(5,5,np.shape(x)[1],0.1,0.8,5000,cosine_distance)
    som.train(x)
    som.plot(x,y)
    time_2 = datetime.now()-start_2
    print(time_2)
    
    print("MANHATTAN DISTANCE")
    start_3=datetime.now()
    som = SOM(5,5,np.shape(x)[1],0.1,0.8,5000,manhattan_distance)
    som.train(x)
    som.plot(x,y)
    time_3 = datetime.now()-start_3
    print(time_3)

main()
