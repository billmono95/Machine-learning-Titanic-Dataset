# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 07:34:01 2020

@author: Bill
"""


import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def sigmoid(self,z):
        
        result = 1 / (1 + np.exp(-z))
        return result
    
    
    def predict(self,X, w, b):
    
    
        logits = X @ w + b
        return self.sigmoid(logits)
    
    def cross_entropy(self,Y, P):

        eps = 1e-3
        P = np.clip(P, eps, 1 - eps)  
        return -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean()
    
    
    def plot_loss(self,fignum,title, iterations, loss):
        loss = [x for x in loss if x is not None]
        plt.figure(fignum)
        plt.clf()
        plt.title(title)
        plt.xlabel("Iterations")
        plt.plot(iterations, loss)
        plt.pause(0.0001)
    
    
    def fit(self,X,y, epoch = 1000, learning_rate = 0.005):
        
            m,n = X.shape
            self.w = np.zeros(n)
            self.b = 0
            iterations = []
            loss_values = []
            
            for i in range(epoch):
                iterations.append(i)
                P = self.predict(X,self.w,self.b)
                loss = self.cross_entropy(y,P)
                loss_values.append(loss)
                grad_w = ((P - y) @ X) / m
                grad_b = (P - y).mean()
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b
            #self.plot_loss(1,"Loss function", iterations, loss_values)    
            return self.w,self.b
        
    def fit_l2(self,X,y,lambda_, epoch = 10000, learning_rate = 0.005):
        
            m,n = X.shape
            self.w = np.zeros(n)
            self.b = 0
            
            for i in range(epoch):
                P = self.predict(X,self.w,self.b)
                
                grad_w = ((P - y) @ X) / m  + 2 * lambda_ * self.w
                grad_b = (P - y).mean()
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b
            return self.w,self.b
        
        
    def fit_l1(self,X,y,lambda_, epoch = 10000, learning_rate = 0.005):
        
            m,n = X.shape
            self.w = np.zeros(n)
            self.b = 0
            
            for i in range(epoch):
                P = self.predict(X,self.w,self.b)
                
                grad_w = ((P - y) @ X) / m + lambda_ * np.sign(self.w)
                grad_b = (P - y).mean()
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b
            return self.w,self.b  
        
        
        








        