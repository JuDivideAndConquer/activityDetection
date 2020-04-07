
import random
from math import pi,exp,log,pow,gamma,sin
from cmath import phase
import svm
import read
import numpy as np


def init_population(population_count=20,dim=516):
    population=list()
    for i in range(population_count):
        x=list()
        for j in range(dim):
            x.append(random.uniform(0, 1))
        population.append(x)
    return population
'''def init_population(count=20,tot_fts=561):
    init_popul=list()
    j=0
    while(j<count):
        new=list()
        for i in range(tot_fts):
            bit=random.randint(0,1)
            new.append(bit)
        if (new not in init_popul):
            init_popul.append(new)
            j+=1
                
    return init_popul'''


def extractFeatures(x_train,x_test,feature_map):
    x_train_cur=list()
    x_test_cur=list()

    index_list = [i for i, val in enumerate(feature_map) if val]
    #print("INDEX LIST:",np.asarray(index_list).shape)


    #making new train set with new features
    for j in range(len(x_train)):
        res_list = [x_train[j][i] for i in index_list]
        x_train_cur.append(res_list)
    #print("X_TRAIN_CUR:",np.asarray(x_train_cur).shape)


        
    #print(len(res_list[0]))
    for j in range(len(x_test)):
        res_list = [x_test[j][i] for i in index_list]
        x_test_cur.append(res_list)
    #print("X_TEST_CUR:",np.asarray(x_test_cur).shape)
    return x_train_cur,x_test_cur


def returnAccuracy(x_train_cur,x_test_cur,y_train,y_test):
    #training-testing 
    accuracy=svm.SVM(x_train_cur,y_train,x_test_cur,y_test)
    return accuracy

def returnAccuracyList(count,x_train,x_test,y_train,y_test,feature_map):
    accuracy_list=list()
    for i in range(count):
        #print("FEATURE MAP :",i)
        x_train_cur,x_test_cur=extractFeatures(x_train,x_test,feature_map[i])
        accuracy=returnAccuracy(x_train_cur,x_test_cur,y_train,y_test)
        accuracy_list.append(accuracy)
    return accuracy_list


#----------------------------------------------------make features--------------------------------------------------------------

def make_feature(population,population_count,dim):
    result=list()
    for i in range(population_count):
        x=list()
        for j in range(dim):
            if(population[i][j]>=0.5):
                x.append(1)
            else:
                x.append(0)
        result.append(x)
    return result



iteration=0
x_train=list()
y_train=list()
x_test=list()
y_test=list()
accuracy_list=list()
population_count=20
dim=516
count=20

#reading training/testing datasets
column_names,x_train,y_train,train_count=read.read('../Data/1/train.csv')
column_names,x_test,y_test,test_count=read.read('../Data/1/test.csv')
        
population_old=init_population()
print("OLD_POPULATION:",np.asarray(population_old).shape)
        
#population_new=geneticAlgo(tot_fts_count,population_old,count)
#print("NEW_POPULATION:",np.asarray(population_new).shape)


print("ITERATION :",iteration)
feature_map=make_feature(population_old,population_count,dim)
#print(population_old)
#print(feature_map)
accuracy_list=returnAccuracyList(count,x_train,x_test,y_train,y_test,feature_map)

iteration+=1
        
while(iteration<15):
    print("ITERATION :",iteration)
    print("NEW_POPULATION:",np.asarray(population_new).shape)
    accuracy_list_new=returnAccuracyList(count,x_train,x_test,y_train,y_test,population_new)
    print (np.asarray(population_old).shape,np.asarray(accuracy_list).shape)
    iteration+=1