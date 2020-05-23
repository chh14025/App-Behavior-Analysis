#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import seaborn as sns
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

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1',solver='liblinear')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)


df_cm = pd.DataFrame(cm, index = (0,1), columns = (0,1))
sns.heatmap(df_cm, annot = True, fmt='g')


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print ('Logistic Accuracy: %0.3f(+/- %0.3f)' %(accuracies.mean(), accuracies.std()*2))

final_results = pd.concat([y_test, test_identifier], axis = 1).dropna()
final_results['predicted_results'] = y_pred
final_results[['user', 'enrolled','predicted_results']].reset_index(drop = True)
