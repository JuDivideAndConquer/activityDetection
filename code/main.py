import csv
import importlib
import read
#our program files
import svm
import geneticAlgo

#data structures 
column_names=list()
train_count=0
test_count=0
x_train=list()
y_train=list()
x_test=list()
y_test=list()
init_population1=[1]*561 #used as a feature map
x=[1]*280
y=[0]*281
x.extend(y)
init_population2=list(x) #used as a feature map
feature_id=0

#reading training/testing datasets
column_names,x_train,y_train,train_count=read.read('../Data/1/train.csv')
column_names,x_test,y_test,test_count=read.read('../Data/1/test.csv')

#feature selection
#GENETIC ALGORITHM
cur_population=list()
cur_population.append(init_population1)
cur_population.append(init_population2)
cur_count=len(cur_population)

with open('../Result/svm.csv',mode='w') as result_file:
        result_writer=csv.writer(result_file)
        l=list()
        l.append("ID")
        l.append("Feature Map")
        l.append("Params")
        l.append("Accuracy")
        result_writer.writerow(l)

while 1:
    print("Forming new population")
    new_population,cur_population=geneticAlgo.geneticAlgo(561,cur_population,cur_count)
    print("New Population formed")

    for member in new_population:
        x_train_cur=list()
        x_test_cur=list()
        index_list = [i for i, val in enumerate(member) if val]
        #making new train set with new features
        for j in range(len(x_train)):
            res_list = [x_train[i] for i in index_list]
            x_train_cur.append(res_list)
        for j in range(len(x_test)):
            res_list = [x_test[i] for i in index_list]
            x_test_cur.append(res_list)
        #training- testing 
        accuracy,params=svm.SVM(x_train,y_train,x_test,y_test)
        #saving the result
        with open('../Result/svm.csv',mode='a+') as result_file:
            result_writer=csv.writer(result_file)
            l=list()
            l.append(id)
            l.append(member)
            l.append(params)
            l.append(accuracy)
            result_writer.writerow(l)
            id+=1
    cur_population=cur_population.extend(new_population)
    cur_count=len(cur_population)
    
    break






