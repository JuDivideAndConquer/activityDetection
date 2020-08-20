import random
from math import pi,log,pow,gamma,sin,sqrt,inf,exp
from cmath import phase
import svm
import read
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
from numpy import exp
import pandas as pd
import time,sys

omega=0.9
lamda=0.5
alfa=0.99


#----------------------------------sigmoid function-----------------------------------------------s

def sigmoid(gamma):
	#print(gamma)
	if gamma < 0:
		return 1 - 1/(1 + exp(gamma))
	else:
		return 1/(1 + exp(-gamma))


#------------------------result between 0-1-------------------------------------------------------
def population_for_f(population,population_count,dim):
	temp=list()
	for i in range(population_count):
		x=list()
		for j in range(dim):
			x.append(sigmoid(population[i][j]))
		temp.append(x)
	return temp

#---------------------------------------------1D initilization-----------------------------------------------------
def intz(population,N1,dim):
	x=list()
	for j in range(dim):
		p=random.randrange(N1)
		d=random.randrange(dim)
		x.append(population[p][d])
	return x

#--------------------------------------initilize the population----------------------------------------------------

def init_population(population_count,dim):
	population=list()
	for i in range(population_count):
		x=list()
		for j in range(dim):
			x.append(random.uniform(0,1))
		population.append(x)
	return population

#-------------------------------------calculate theta-------------------------------------------------------------

def calculate_theta(g,N):
	theta=0.5*(1+g/N)
	return theta

#-----------------------------------update population--------------------------------------------------------------

def update_population(theta,population,dim,N1):
	result=list()
	for i in range(N1):
		x=list()
		for d in range(dim):
			temp=population[i][d]/theta+random.uniform(0,1)*population[i][d]
			x.append(temp)
		result.append(x)
	return result


#-----------------------------------------calculate value of d--------------------------------------------------------

def calculate_d(N1,N2,population,dim,it):
	result=list()
	sum=0
	for d in range(dim):
		for i in range(N1+N2):
			sum+=(population[it][d]-population[i][d])**2
		sum=sum**0.5
		result.append(sum)
	return result

#--------------------------------update population in step 2---------------------------------------------------------

def update_population1(i,y,dim,population):
	result=list()
	for d in range(dim):
		temp=population[i][d]+alfa*y
		result.append(temp)
	return result

def step2_upadte(N1,N2,population,dim):
	result=list()
	for i in range(N1+N2):
		distance=calculate_d(N1,N2,population,dim,i)
		distance1=distance
		distance.sort()
		for d in range(dim):
			if(distance1[d]==distance[0]):
				x1=population[i][d]
			if(distance1[d]==distance[1]):
				x2=population[i][d]
		y=lamda*x1+(1-lamda)*x2
		pop=update_population1(i,y,dim,population)
		result.append(pop)
	return result
#------------------------------------------------------------step4 update-----------------------------------------------

def mask(pop1,pop2,dim):
	res=list()
	for d in range(dim):
		bit=random.randint(0,1)
		if(bit==0):
			res.append(pop1[d])
		else:
			res.append(pop2[d])
	return res


def step4_update(population,N1,dim,N4):
	result=list()
	for k in range(N4):
		S=intz(population,N1,dim)
		r=random.randrange(N1)
		T=population[r]
		pop=mask(S,T,dim)
		result.append(pop)
	return result




#---------------------------------------features for 1D-----------------------------------------------------------

def features_1D(pop,dim):
	x=list()
	sum=0
	di=1/dim
	for d in range(dim):
		sum+=pop[d]
	for d in range(dim):
		temp=pop[d]/sum
		if(temp>=di):
			x.append(1)
		else:
			x.append(0)
	return x


#-------------------------------------make features-------------------------------------------------------------------

def make_feature(population,population_count,dim):
    result=list()
    di=1/dim
    for i in range(population_count):
    	x=features_1D(population[i],dim)
    	result.append(x)
    return result

#-------------------------------extracting fetures-------------------------------------------------

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


#-----------------------------------------------fitness calculation in 1D--------------------------------------------

def fitness_1D(feature,dim,x_train_cur,x_test_cur,y_train,y_test):
	accuracy=svm.SVM(x_train_cur,y_train,x_test_cur,y_test)
	set_count=sum(feature)/dim
	set_cnt=1-set_count
	val=accuracy*omega+set_cnt*(1-omega)
	#print(set_count,"   ",accuracy,"  ",val)
	return val,accuracy

#--------------------------------------------calculate the fitnes-----------------------------------------------------

def fitness(feature_map,population_count,dim,x_train,x_test,y_train,y_test):
	result=list()
	for i in range(population_count):
		x_train_cur,x_test_cur=extractFeatures(x_train,x_test,feature_map[i])
		fit,accuracy=fitness_1D(feature_map[i],dim,x_train_cur,x_test_cur,y_train,y_test)
		result.append(fit)
	return result

#---------------------------------------------------save the results in a file-----------------------------------------

def saveInCSV_mini(feature_id,accuracy,population_count,datasetname):
	fname='../result1/'+datasetname+'.csv'
	with open(fname,mode='a+') as result_file:
		result_writer=csv.writer(result_file)
		l=list()
		if(feature_id!=-1):
			l.append(feature_id)
		l.append(accuracy)
		result_writer.writerow(l)

#------------------------------------------check result-----------------------------------------------------------------

def check_result(feature_map,population_count,dim,x_train,x_test,y_train,y_test):
	fitX=list()
	accuracy_list=list()
	for i in range(population_count):
		x_train_cur,x_test_cur=extractFeatures(x_train,x_test,feature_map[i])
		fit,accuracy=fitness_1D(feature_map[i],dim,x_train_cur,x_test_cur,y_train,y_test)
		fitX.append(fit)
		accuracy_list.append(accuracy)
	print("No.of Features: ",sum(feature_map[0]),"Accuracy: ",accuracy_list[0],"FitNess: ",fitX[0])
	saveInCSV_mini(sum(feature_map[0]),accuracy_list[0],population_count,datasetname)


def improved_tree(datasetname):
	MAX_ITER=100
	g=0
	N1=6
	N2=2
	N3=2
	N4=5
	population_count=10
	dataset='../Data/UCI_DATA-master/'+datasetname+'/'+datasetname+'.csv'
	df = pd.read_csv(dataset)
	a, b = np.shape(df)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	dim = data.shape[1]
	cross = 5
	test_size = (1/cross)
	x_train,x_test,y_train,y_test = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
	print(datasetname)
	print(len(x_train[0]))
	population=init_population(population_count,dim)
	feature_map=make_feature(population_for_f(population,population_count,dim),population_count,dim)
	fitlist=fitness(feature_map,population_count,dim,x_train,x_test,y_train,y_test)
	fitlist_res,population_res=zip(*sorted(zip(fitlist,population),reverse=True))
	population=list(population_res)
	fitlist=list(fitlist_res)
	check_result(feature_map,population_count,dim,x_train,x_test,y_train,y_test)
	while(g<MAX_ITER):
		g=g+1
		print("Iteration: ",g)

		#==============================================================================================
		theta=calculate_theta(g,MAX_ITER)
		population_new=update_population(theta,population,dim,N1)
		feature_map=make_feature(population_for_f(population_new,N1,dim),N1,dim)
		fitlist_new=fitness(feature_map,N1,dim,x_train,x_test,y_train,y_test)
		pop=population[0:N1]
		fitn=fitlist[0:N1]
		population_new.extend(pop)
		fitlist_new.extend(fitn)
		fitlist_res,population_res=zip(*sorted(zip(fitlist_new,population_new),reverse=True))
		for i in range(N1):
			if(fitlist[i]<fitlist_res[i]):
				fitlist[i]=fitlist_res[i]
				population[i]=population_res[i]
		#===============================================================================================
		population_new=step2_upadte(N1,N2,population,dim)
		feature_map=make_feature(population_for_f(population_new,N2,dim),N2,dim)
		fitlist_new=fitness(feature_map,N2,dim,x_train,x_test,y_train,y_test)
		pop=population[N1:N1+N2]
		fitn=fitlist[N1:N1+N2]
		population_new.extend(pop)
		fitlist_new.extend(fitn)
		fitlist_res,population_res=zip(*sorted(zip(fitlist_new,population_new),reverse=True))
		for i in range(N1,N1+N2):
			if(fitlist[i]<fitlist_res[i-N1]):
				fitlist[i]=fitlist_res[i-N1]
				population[i]=population_res[i-N1]
		#===============================================================================================
		population_new=init_population(N3,dim)
		feature_map=make_feature(population_for_f(population,N3,dim),N3,dim)
		fitlist_new=fitness(feature_map,N3,dim,x_train,x_test,y_train,y_test)
		for i in range(N1+N2,population_count):
			population[i]=population_new[i-N1-N2]
			fitlist[i]=fitlist_new[i-N1-N2]
		#===============================================================================================
		population_new=step4_update(population,N1,dim,N4)
		feature_map=make_feature(population_for_f(population_new,N4,dim),N4,dim)
		fitlist_new=fitness(feature_map,N4,dim,x_train,x_test,y_train,y_test)
		fitlist.extend(fitlist_new)
		population.extend(population_new)
		fitlist_res,population_res=zip(*sorted(zip(fitlist,population),reverse=True))
		#fitlist=list(fitlist_res[0:10])
		#population=list(population_res[0:10])
		for i in range(population_count):
			if(fitlist_res[i]>fitlist[i]):
				fitlist[i]=fitlist_res[i]
				population[i]=population_res[i]
		feature_map=make_feature(population_for_f(population,population_count,dim),population_count,dim)
		check_result(feature_map,population_count,dim,x_train,x_test,y_train,y_test)


datasetlist = ["BreastCancer", "BreastEW", "CongressEW", "Exactly", "Exactly2", "HeartEW", "Ionosphere",  "Lymphography"]
#datasetlist = ["M-of-n", "PenglungEW", "Sonar", "SpectEW", "Tic-tac-toe", "Vote", "Wine", "Zoo"]
for datasetname  in datasetlist:
	improved_tree(datasetname)




