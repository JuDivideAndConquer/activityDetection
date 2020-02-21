import csv
import importlib


#data structures 
column_names=list()
train_count=0
test_count=0
x_train=list()
y_train=list()
x_test=list()
y_test=list()

#reading training/testing datasets
import read
column_names,x_train,y_train,train_count=read.read('../Data/1/train.csv')
column_names,x_test,y_test,test_count=read.read('../Data/1/test.csv')

import svm
print (svm.SVM(x_train,y_train,x_test,y_test))