#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:27:41 2020

@author: s.p.
"""


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

dataset = pd.read_csv('new_appdata10.csv')

response = dataset['enrolled']
dataset = dataset.drop(columns = 'enrolled')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset,response,test_size=0.2, random_state = 0)

train_identifier = x_train ['user']
x_train = x_train.drop(columns = 'user')
test_identifier = x_test ['user']
x_test = x_test.drop(columns = 'user')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = pd.DataFrame(sc.fit_transform(x_train))
x_test2 = pd.DataFrame(sc.transform(x_test))
x_train2.columns = x_train.columns.values
x_test2.columns = x_test.columns.values
x_train2.index = x_train.index.values
x_test2.index = x_test.index.values

x_train = x_train2
x_test = x_test2
