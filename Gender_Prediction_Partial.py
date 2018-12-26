# -*- coding: utf-8 -*-
"""
Created on Sunday: Jul 15 2018

@author: Yifan Peng for Lookalike Project using GBDT (partial)
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

x_train=pd.read_csv('/home/hadoop/sdl/hdfs_data/64/X_train.csv_171')
y_train=pd.read_csv('/home/hadoop/sdl/hdfs_data/64/y_train.csv_551', header=None)
xtest=pd.read_csv('/home/hadoop/sdl/hdfs_data/64/X_test.csv_734')

x_train = x_train.drop('tdid', axis=1)
y_train.columns = ['gender']
xtest_f=xtest.drop(['tdid'],axis=1)
X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(x_train,y_train,test_size=0.3)

#step1: parameter tuning for n_estimators,
##n_estimators #for i in range(10,101,10):
for i in range(10,101,10):
    GBDT=GradientBoostingClassifier(learning_rate=0.1, n_estimators=i)
    GBDT.fit(X_Train, Y_Train)
    x_train_pred=GBDT.predict(X_Train)
    x_val_pred=GBDT.predict(X_Test)
    (train_score, val_score,auc_score)= (accuracy_score(Y_Train, x_train_pred),accuracy_score(Y_Test, x_val_pred),roc_auc_score(Y_Test, x_val_pred))
    print("GBDT:", ", n_estimators:",i,", Train Accuracy:", train_score, ", Validation Accuracy:", val_score, ",AUC score",auc_score)
 
#step1: parameter tuning for n_estimators,
##n_estimators #for i in range(10,101,10):
for i in range(100,201,10):
    GBDT=GradientBoostingClassifier(learning_rate=0.1, n_estimators=i)
    GBDT.fit(X_Train, Y_Train)
    x_train_pred=GBDT.predict(X_Train)
    x_val_pred=GBDT.predict(X_Test)   
    (train_score, val_score,auc_score)= (accuracy_score(Y_Train, x_train_pred),accuracy_score(Y_Test, x_val_pred),roc_auc_score(Y_Test, x_val_pred))
    print("GBDT:", ", n_estimators:",i,", Train Accuracy:", train_score, ", Validation Accuracy:", val_score, ",AUC score",auc_score)

#step2: parameter tuning for learning_rate,
##learning_rate #for l in range(0.1,0.9,0.1):

for l in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    GBDT=GradientBoostingClassifier(learning_rate=l, n_estimators=200)
    GBDT.fit(X_Train, Y_Train)
    y_train_pred=GBDT.predict(X_Train)
    y_val_pred=GBDT.predict(X_Test)
    y_val_prob=GBDT.predict_proba(X_Test)[:,1]
    (train_score, val_score,auc_score)= (accuracy_score(Y_Train, y_train_pred),accuracy_score(Y_Test, y_val_pred),roc_auc_score(Y_Test, y_val_prob))
    print("GBDT:", ", n_estimators:",l,", Train Accuracy:", train_score, ", Validation Accuracy:", val_score, ",AUC score",auc_score)
    
#step2: parameter tuning learning_rate,
##learning_rate #for l in range(0.31,0.039,0.02):

for l in [0.31, 0.33, 0.35, 0.37, 0.39]:
    GBDT=GradientBoostingClassifier(learning_rate=l, n_estimators=200)
    GBDT.fit(X_Train, Y_Train)
    y_train_pred=GBDT.predict(X_Train)
    y_val_pred=GBDT.predict(X_Test)
    y_val_prob=GBDT.predict_proba(X_Test)[:,1]
    (train_score, val_score,auc_score)= (accuracy_score(Y_Train, y_train_pred),accuracy_score(Y_Test, y_val_pred),roc_auc_score(Y_Test, y_val_prob))
    print("GBDT:", ", n_estimators:",l,", Train Accuracy:", train_score, ", Validation Accuracy:", val_score, ",AUC score",auc_score)
 
GBDT=GradientBoostingClassifier(learning_rate=0.31, n_estimators=200,min_samples_split=121,min_samples_leaf=55)
GBDT.fit(X_Train, Y_Train)
y_train_pred=GBDT.predict(X_Train)
y_val_pred=GBDT.predict(X_Test)
y_val_prob=GBDT.predict_proba(X_Test)[:,1]        
(train_score, val_score,auc_score)= (accuracy_score(Y_Train, y_train_pred),accuracy_score(Y_Test, y_val_pred),roc_auc_score(Y_Test, y_val_prob))
print("GBDT:", ", min_samples_leaf ",", Train Accuracy:", train_score, ", Validation Accuracy:", val_score, ",AUC score",auc_score)

'''
('GBDT:', ', min_samples_leaf ', ', Train Accuracy:', 0.73216317243620999, ', Validation Accuracy:', 0.7151475455046884, ',AUC score', 0.7781745835706062)

'''
train_p=GBDT.predict_proba(xtest_f)[:,1]
train_p
'''
array([ 0.76389614,  0.66847588,  0.67199406, ...,  0.71863811,
        0.68279139,  0.78397069])
'''
train_p_f=pd.DataFrame(train_p,columns=['probability'])
df=pd.concat([xtest['tdid'],train_p_f],axis=1)
df.to_csv('/home/hadoop/sdl/hdfs_data/gender_prediction/pengyifan.csv')