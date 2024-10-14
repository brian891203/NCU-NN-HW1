# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 03:32:40 2023

@author: tony
"""

import numpy as np


class Perceptron:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.weights = None

    def set_data(self, data):
        self.train_x, self.train_y = data[0], data[1]
        self.test_x, self.test_y = data[2], data[3]
        # self.dim_3 = data[4]
        
    def set_model(self, input_dim, hidden_dim, output_dim):
       # np.random.seed(42)
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim
       self.output_dim = output_dim
       self.num_layers = 2
       self.weights = {
           'W1': np.random.randn(hidden_dim, input_dim),
           'W2': np.random.randn(output_dim, hidden_dim)
       }
       self.biases = {
           'b1': np.zeros(hidden_dim),
           'b2': np.zeros(output_dim)
       }
       # print(self.weights,self.biases)
    
    def forward(self, x):
        # 前向传播
        # print(self.weights['W1'], x, self.biases['b1'])
        z1 = np.dot(self.weights['W1'], x) + self.biases['b1']
        # print(f"z1={z1}")
        self.a1 = self.sigmoid(z1)
        # print(f"a1={self.a1}")
        # print(self.weights['W2'],f"a1={self.a1}",self.biases['b2'])
        z2 = np.dot(self.weights['W2'], self.a1) + self.biases['b2']
        # print(f"z2={z2}")
        a2 = self.softmax(z2)
        # print(f"a2={a2}")
        return a2
    
    def backward(self, x, y_true, y_pred):
        # 反向传播
        # print(y_true, y_pred)
        loss = self.categorical_cross_entropy(y_true, y_pred)
        dz2 = y_pred - y_true
        # print(f"dz2={dz2}")
        dW2 = np.outer(dz2, self.a1)
        db2 = dz2
        # print(f"dW2={dW2}")
        # print(self.weights['W2'].T, dz2)
        # print(np.dot(self.weights['W2'].T, dz2), self.a1)
        dz1 = np.dot(self.weights['W2'].T, dz2) * self.a1 * (1 - self.a1)
        # print(f"dz1={dz1}",x)
        dW1 = np.outer(dz1, x)
        # print(f"dW1={dW1}")
        db1 = dz1
        return loss, {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    def train(self, epochs=100, lr=0.1):
        self.loss_record = []
        for epoch in range(1,epochs+1):
            total_loss = 0
            # print(f'epoch:{epoch+1}')
            for i in range(self.train_x.shape[0]):
                y_pred = self.forward(self.train_x[i])
                loss, gradients = self.backward(self.train_x[i], self.train_y[i], y_pred)
                total_loss += loss
                for param in self.weights.keys():
                    self.weights[param] -= lr * gradients[param]
                for param in self.biases.keys():
                    self.biases[param] -= lr * gradients[param]
                total_loss += loss
            total_loss /= self.train_x.shape[0]
            self.loss_record.append(total_loss)
            # if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    def evaluate(self, test_x, test_y):
        y_pred=[]
        test_y_ed=[]
        for i in range(test_x.shape[0]):
            # print(test_y[i])
            y_pred_soft = self.forward(test_x[i])
            y_pred_ont_hot = np.argmax(y_pred_soft)
            y_pred.append(y_pred_ont_hot)
        y_pred=np.array(y_pred)
        for i in range(test_y.shape[0]):
            test_y_ed.append(np.argmax(test_y[i]))
        test_y_ed=np.array(test_y_ed)
        # print(y_pred,test_y_ed)
        # test_y[i]
        # correct_predictions = np.sum(np.all(np.equal(y_pred, test_y), axis=1))
        # total_samples = test_x.shape[0]
        # accuracy = correct_predictions / total_samples
        
        # y_pred = np.array(y_pred)
        accuracy = np.mean(test_y_ed == y_pred)
        print("结果(accuracy)：", accuracy)
        # print("结果(accuracy)：", predictions)
        # print(self.biases[f'b{1}'])
        return accuracy
    
    
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def categorical_cross_entropy(self, y_true, y_pred):
        # print(y_true,y_pred)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
