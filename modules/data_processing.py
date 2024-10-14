# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 03:12:10 2023

@author: tony
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataProcessor:
    def load_data(self, file_path, PCA_enable=True):
        #loading data
        # file_path = 'C:/Users/tony/Downloads/NN_HW1_DataSet/NN_HW1_DataSet/basic/2Ccircle1.txt'
        encoder = OneHotEncoder(sparse_output =False)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = np.loadtxt(file)
                # X = data[:,0:2]
                # y = data[:,2].astype(int)
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
        except Exception as e:
            print(f"發生錯誤：{str(e)}")
        dim_3=False
        pca = PCA(n_components=2)
        #dataset<4:test=train
        if data.shape[0]>4:
            #split dataset
            split_ratio = 2/3
            train, test = train_test_split(data, train_size=split_ratio)
            if data.shape[1]>=4:
                dim_3=True
                
                
                # print("3D")
                train_x = train[:,0:data.shape[1]-1]
                train_y = encoder.fit_transform(train[:,-1].astype(int).reshape(-1, 1))
                
                test_x = test[:,0:data.shape[1]-1]
                test_y = encoder.fit_transform(test[:,-1].astype(int).reshape(-1, 1))
                pca.fit(train_x)
                if PCA_enable:
                    train_x = pca.transform(train_x)
                    test_x = pca.transform(test_x)
                print(train_x.shape)
            else:
                pca = PCA(n_components=2)
                train_x = train[:,0:data.shape[1]-1]
                train_y = encoder.fit_transform(train[:,-1].astype(int).reshape(-1, 1))
                
                test_x = test[:,0:data.shape[1]-1]
                test_y = encoder.fit_transform(test[:,-1].astype(int).reshape(-1, 1))
                # print(train_x.shape,test_x)
                pca.fit(train_x)
                if PCA_enable:
                    train_x = pca.transform(train_x)
                    test_x = pca.transform(test_x)
                print(train_x.shape)
        else:
            # print(data.shape[1])
            if data.shape[1]==4:
                dim_3=True
                train_x = data[:,0:data.shape[1]-2]
                train_y = encoder.fit_transform(data[:,-1].astype(int).reshape(-1, 1))
                
                test_x = train_x
                test_y = train_y
                # print(train_y)
                pca.fit(train_x)
                if PCA_enable:
                    train_x = pca.transform(train_x)
                    test_x = pca.transform(test_x)
                print(train_x.shape)
            else:
                train_x = data[:,0:data.shape[1]-1]
                train_y = encoder.fit_transform(data[:,-1].astype(int).reshape(-1, 1))
                
                test_x = train_x
                test_y = train_y
                # print(train_y)
                pca.fit(train_x)
                if PCA_enable:
                    train_x = pca.transform(train_x)
                    test_x = pca.transform(test_x)
                print(train_x.shape)
        return train_x, train_y, test_x, test_y
        