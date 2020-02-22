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

#training- testing 
import svm
accuracy,params=svm.SVM(x_train,y_train,x_test,y_test)

#saving the result
with open('Result/svm.csv',mode='a') as result_file:
    result_writer=csv.writer(result_file)
    fts=list()
    for i in range(len(x_train[0])):
        fts.append(1)
    l=list()
    l.aapend(fts)
    l.append(params)
    l.append(accuracy)
    result_writer.writerow(l)
