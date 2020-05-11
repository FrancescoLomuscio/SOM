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
import pickle
import os



class SOM:
    def __init__(self,x,y,input_dim,learning_rate,sigma,max_iter,
                 decay_function=learning_rate_decay):
        #Initialize class variables
        self._x = x
        self._y = y
        self._input_dim = input_dim
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._max_iter = max_iter
        self._decay_function = decay_function
        
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
        
    def update(self,x,t,max_iter):
        lr = self.decay(self._learning_rate,t,max_iter)
        self._weights += lr*self._weights
        return self._weights
        
        
        
def main():
    som = SOM(1,1,4,0.1,0.5,500)
    x = np.random.rand(1,4)
    a = som.choose_winner(x)
    print(som._weights)
    for i in range(10):
        b = som.update(x,i,som._max_iter)
        print(b)
 


main()
