# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:25:04 2018

@author: joshu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Startup_Data_Set.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values #last column contains the dependent variable

#encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:,3] = le.fit_transform(x[:,3])
ohe = OneHotEncoder(categorical_features = [3])
x = ohe.fit_transform(x).toarray()

#avoiding the dummy variable trap
x = x[:,1:] #removes the first column

#splitting into a training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .2, random_state = 0) #no need for random state. Only used to get the same results
