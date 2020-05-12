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

from sklearn.datasets import load_iris

def _build_iteration_indexes(data_len, num_iterations, random_generator=None):
    iterations = np.arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    return iterations

class SOM:
    def __init__(self,x,y,input_dim,learning_rate,sigma,max_iter):
        
        self._random_generator = np.random.RandomState(42)
        #Initialize class variables
        self._x = x
        self._y = y
        self._input_dim = input_dim
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._max_iter = max_iter
        
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
    
    def euclidean_distance(self,x, w):
        return norm(np.subtract(x,w), axis = -1)
    
    def activate(self, x):
        self._activation_map = self.euclidean_distance(x, self._weights)
    
    def choose_winner(self,x):
        self.activate(x)
        return np.unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)
        
    def update(self,x,win,t,max_iter):
        lr = self.decay(self._learning_rate,t,max_iter)
        sigma = self.decay(self._sigma,t,max_iter)
        
        g = self._gaussian(win, sigma)*lr
        
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        return self._weights
    
    def _gaussian(self, c, sigma):
        d = 2*np.pi*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T 
    
        
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
        colors = ['#EDB233', '#90C3EC', '#C02942', '#79BD9A', '#774F38', 'gray', 'black']
        markers = ['o'] * 3
        
        wm = self.winner_map(data)
        fig, ax = plt.subplots(figsize=(self._x,self._y))
        plt.pcolormesh(wm, edgecolors=None)
        #plt.colorbar()
        plt.xticks(np.arange(.5, self._x + .5), range(self._x))
        plt.yticks(np.arange(.5, self._y + .5), range(self._y))
        #ax.set_aspect('equal')

        #fig, ax = plt.subplots(figsize=(5,5))
        for cnt, xx in enumerate(data):
            c = colors[targets[cnt]]
            w = self.choose_winner(xx)
            ax.plot(w[1] + .5 + 0.1 * np.random.randn(1), w[0] + .5 + 0.1 * np.random.randn(1),
                    markers[targets[cnt]], color=c, markersize=12)
        
        plt.show()
        
        plt.plot(self._history)
        
                
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
    som = SOM(10,10,4,0.1,0.8,10000)
    iris = load_iris()
    x = iris.data[:, :]  
    y = iris.target
    som.train(x)
    som.plot(x,y)

main()
