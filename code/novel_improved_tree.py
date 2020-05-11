import random
from math import pi,log,pow,gamma,sin,sqrt,inf
from cmath import phase
import svm
import read
import numpy as np
from sklearn.model_selection import train_test_split
import importlib
import csv
from scipy.stats import norm
from numpy import exp

#----------------------------------sigmoid fuhnction-----------------------------------------------s

def sigmoid(x):
	return exp(x)/(1+exp(x))
	'''finally:
		#print(x)
		x=round(x)
		ex=exp(-1)
		p=float(ex**x)
		p+=1
		return 1/p'''


#-------------------------calculate theta--------------------------------------

def calculate_theta(g,G):
	theta=0.5*(1+(3*g)/G)
	return theta


#------------------------calculate d--------------------------------------------

def claculate_d(N1,N2,population,dim):
	result=list()
	sum=0
	for d in range(dim):
		for i in range(N1+N2):
			sum+=(population[N2-1][d]-population[i][d])**2
		sum=sum**0.5
		if(N2==i):
			sum=inf
		result.append(sum)
	return result




#--------------------------init_the population---------------------------------

def init_population(population_count,dim):
	population=list()
	for i in range(population_count):
		x=list()
		for j in range(dim):
			x.append(random.uniform(0, 2))
		population.append(x)
	return population

#-----------------------------update population---------------------------------------

def update_population(dim,population,theta,i):
	r=random.uniform(0,1)
	result=list()
	for d in range(dim):
		temp=population[i][d]/theta+r*population[i][d]
		result.append(temp)
	return result

#--------------------------another update population-------------------------------------

def update_population1(i,alfa,y):
	result=list()
	for d in range(dim):
		temp=population[i][d]+alfa*y
		result.append(temp)
	return result


#-------------------------------make features--------------------------------------------------

def make_feature(population,population_count,dim):
    result=list()
    #print(temp)
    for i in range(population_count):
    	x=list()
    	for j in range(dim):
    		if(abs(population[i][j])>=0.5):
    			x.append(1)
    		else:
    			x.append(0)
    	result.append(x)
    return result

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

#------------------------result between 0-1-------------------------------------------------------
def population_for_f(population,population_count,dim):
	temp=list()
	for i in range(population_count):
		x=list()
		for j in range(dim):
			x.append(sigmoid(population[i][j]))
		temp.append(x)
	return temp


#--------------------------------------------main fun-----------------------------

Max_iter=50
g=0
population_count=10
dim=29
population=init_population(population_count,dim)
#reading training/testing datasets
column_names,x,y,train_count=read.read('../Data/UCI_DATA-master/BreastEW/BreastEW.csv')
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.20, random_state=1)
print(len(x_train[0]))

feature_map=make_feature(population,population_count,dim)
#print(len(x[0]))
print("OLD_POPULATION:",np.asarray(population).shape)
accuracy_list=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map)
#print(accuracy_list)
accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))
population=list(population_res[0:20])
accuracy_list=list(accuracy_res[0:20])
#print(accuracy_res)
population_new=population;
N1=6
N2=2
N3=2
lemda=0.5
alfa=0.99
#saveInCSV(g,population,accuracy_list)
#print(feature_map)
while(g<Max_iter):
	g=g+1
	print("ITERATION :",g)
	theta=calculate_theta(g,Max_iter)
	for i in range(N1):
		population_new[i]=update_population(dim,population,theta,i)
	feature_map=make_feature(population_for_f(population_new,N1,dim),N1,dim)
	#print(feature_map)
	for i in range(N1):
		x_train_cur,x_test_cur=extractFeatures(x_train,x_test,feature_map[i])
		accuracy=returnAccuracy(x_train_cur,x_test_cur,y_train,y_test)
		if(accuracy>accuracy_list[i]):
			population[i]=population_new[i]
			accuracy_list[i]=accuracy
	for i in range(N1,N1+N2):
		distance=claculate_d(N1,N2,population,dim)
		distance1=distance
		distance.sort()
		for d in range(dim):
			if(distance1[d]==distance[0]):
				x1=population[i][d]
			if(distance1[d]==distance[1]):
				x2=population[i][d]
		y=lemda*x1+(1-lemda)*x2
		population_new[i-N1]=update_population1(N2,alfa,y)
	feature_map=make_feature(population_for_f(population_new,N2,dim),N2,dim)
	for i in range(0,N2):
		x_train_cur,x_test_cur=extractFeatures(x_train,x_test,feature_map[i])
		accuracy=returnAccuracy(x_train_cur,x_test_cur,y_train,y_test)
		if(accuracy>accuracy_list[i+N1]):
			population[i+N1]=population_new[i]
			accuracy_list[i+N1]=accuracy
	for i in range(0,population_count):
		population_new[i]=update_population(dim,population,theta,i)
	#==========================================================================================================
	feature_map=make_feature(population_for_f(population_new,population_count,dim),population_count,dim)
	#print(feature_map)
	print("After Fission:",np.asarray(population).shape)
	accuracy_list_new=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map)
	
	population.extend(population_new)
	accuracy_list.extend(accuracy_list_new)

	accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))

	population=list(population_res[0:5])
	accuracy_list=list(accuracy_res[0:5])
	print (np.asarray(population).shape,np.asarray(accuracy_list).shape)
	#=========================================================================================================












