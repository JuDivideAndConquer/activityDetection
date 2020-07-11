
import random
from math import pi,log,pow,gamma,sin,sqrt,exp
from cmath import phase
import knn
import svm
import read
import numpy as np
import time,sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
import pandas as pd
'''
    A variable rand_p determines reaction
'''
P_fi = 0.5 #probabiltiy of fission reaction
P_beta = 0.75 #probability of beta decay
folder="BreastEW"


#--------------------------------------diracdelta fun------------------------------------------

def diracdelta(x):
	elep=.000001
	result=(1/pi)*(elep/(x**2+elep**2))
	#print(result)
	return result

#----------------------------------sigmoid fuhnction-----------------------------------------------s

def sigmoid(gamma):
    #print(gamma)
    if gamma < 0:
        return 1 - 1/(1 + exp(gamma))
    else:
        return 1/(1 + exp(-gamma))

#initializing the population
def init_population(population_count,dim):
    population=list()
    for i in range(population_count):
        x=list()
        for j in range(dim):
            x.append(random.uniform(0, 1))
        population.append(x)
    #print(population)
    return population

#Heated neutron generation
# Nei <- (xi+xj)/2
#xi,xj are nuclei
def neutron_generation(i,j,population,dim):
	while(j==i):
		j=random.randint(0,population_count-1)
	neutron=list()
	for k in range(dim):
		temp=(population[i][k]+population[j][k])/2
		neutron.append(temp)
	return neutron

#theta calculation
#balance factor for exploitation and exploration
def theta_calculation(population,i,dim,g):
    population_count=len(population)
    theta=list()
    res=log(g)/g
    for k in range(dim):
        temp=res*abs(population[i][k]-population[0][k])
        theta.append(temp)
    return theta

def Gaussian_1D(mean,theta,x):
	if(theta==0):
		return diracdelta(x)
	exponent=-0.5*((x-mean)/theta)**2
	sqt=sqrt(2*pi)
	coefficient=1/(theta*sqt)
	result=coefficient*exp(exponent)
	#print(result)
	return result

def Gaussian(mean,theta,x,dim): #dim dimension
    result=list()
    for i in range(dim):
        temp=Gaussian_1D(mean[i],theta[i],x[i])
        result.append(temp)
    #print(result)
    return result
#------------------------------------------------
# #Nuclear Fission Phase (NFi)
#------------------------------------------------
# odd nuclei can generate 2 products SF or PF


def odd_fission_SF(i,population,dim,rand_p,g):
	population_count=len(population)
	j=random.randint(0,population_count-1)
	Nei=neutron_generation(i,j,population,dim)
	Pne=round(rand_p+1)
	theta=theta_calculation(population,i,dim,g)
	gaussian=Gaussian(population[0],theta,population[i],dim)
	result=list()
	for j in range(dim):
		temp=gaussian[j]+(rand_p*population[0][j]-Pne*Nei[j])
		result.append(temp)
	return result

def odd_fission_PF(i,population,dim,rand_p,g):
	j=random.randint(0,population_count-1)
	Nei=neutron_generation(i,j,population,dim)
	Pne=round(rand_p+1)
	r=random.randint(0,len(population)-1)
	while r==i:
		r=random.randint(0,len(population)-1)
	theta=theta_calculation(population,r,dim,g)
	gaussian=Gaussian(population[0],theta,population[i],dim)
	result=list()
	for j in range(dim):
		temp=gaussian[j]+rand_p*population[0][j]-Pne*Nei[j]
		result.append(temp)
	return result
    
def even_fission_no_product(i,population,dim,g):
    theta=theta_calculation(population,i,dim,g)
    gaussian=Gaussian(population[0],theta,population[i],dim)
    result=list()
    for j in range(dim):
        temp=gaussian[j]
        result.append(temp)
    #print(result)
    return result

#------------------------------------------------
# #Ionization stage
#------------------------------------------------

#------------------------------------------eq 10-----------------------------------------
#pa[i]=rank(fitX[i]Fi)/N

def calculate_Pa(population_count):#how to claculate fitX
    Pa=list()
    print(population_count)
    for i in range(population_count):
        rank_of_jth_pop=i#confusion how to claculate the rank function and fitof a popuation is not calculated
        temp=rank_of_jth_pop/population_count
        Pa.append(temp)
    #print(Pa)
    return Pa;

#------------------------------------------------------eq11 &eq 12-------------------------

#X[i][d]ion=X[r1][d]Fi-rand*(X[r2][d]Fi-X[i][d]Fi)...if rand<=0.5
#X[i][d]ion=X[r1][d]Fi-rand*(X[r2][d]Fi-X[i][d]Fi)...if rand>0.5
def ionation_stage1(i,d,popuation,rand):
	r1=random.randint(0,population_count-1)
	r2=random.randint(0,population_count-1)
	if(rand<=0.5):
		return (popuation[r1][d]+rand*(popuation[r2][d]-popuation[i][d]))
	else:
		return (popuation[r1][d]-rand*(popuation[r2][d]-popuation[i][d]))

#----------------------------------------------------eq 13-------------------------------------
#Xion[i][d]=XFi[i][d]+round(rand).rand.(XFi[worst][d]-X[best][d]);
def equ_no_13(i,d,rand_p,population,dim):
	result=population[i][d]+round(rand_p)*rand_p*(population[population_count-1][d]-population[0][d])
	return result



#------------------------------------------------
# #Nuclear Fussion Phase (NFu)
#------------------------------------------------
# odd nuclei can generate 2 products SF or PF


#------------------------------------------eq 14-----------------------------------------
#pc[i]=rank(fitX[i]Ion)/N

def calculate_Pc(N):#how to claculate fitX
    Pc=list()
    for i in range(N):
        rank_of_jth_pop=i;#confusion how to claculate the rank function and fitof a popuation is not calculated
        temp=rank_of_jth_pop/N
        Pc.append(temp)
    return Pc;

#---------------------------------------------eq 15----------------------------------------
#X[j]=X[j]Ion+rand*(X[r1]IOn-X[best]Ion)-e(-norm(Xr1Ion-Xr2Ion)).(Xr1Ion-Xr2Ion)
def fusion_stage1(j,population,rand_p,dim):
	r1=random.randint(0,population_count-1)
	r2=random.randint(0,population_count-1)
	result=list()
	for i in range(dim):
		x=abs(population[r1][i]-population[r2][i])#what is norm based 
		e_pow=0-phase(x)
		temp=population[j][i]+(rand_p*(population[r1][i]-population[0][i]))+(rand_p*(population[r2][i]-population[0][i]))-(exp(e_pow)*population[r1][i]-population[r2][i])
		result.append(temp)
	return result

#----------------------------------------------eq16 &eq17----------------------------------------

def claculate_g(population,k):
    population_count=len(population)
    res=0
    for i in range(population_count):
        res=res+population[k][i]
    return res

#X[j]Fu=X[j]Ion-0.5(sin(2*pi*freq*g+pi).Gmax-g/Gmax +1)(X[r1]ion-X[r1]ion)...if rand>0.5
#X[j]Fu=X[j]Ion-0.5(sin(2*pi*freq*g+pi).g/Gmax +1)(X[r1]ion-X[r1]ion)...if rand<=0.5
def fusion_stage2(i,population,freq,rand_p,dim,g,Gmax):
	r1=random.randint(0,population_count-1)
	r2=random.randint(0,population_count-1)
	result=list()
	for k in range(dim):
		if(rand_p>0.5):
			temp=population[i][k]-0.5*(sin((2*pi*freq*g)+pi)*((Gmax-g)/Gmax)+1)*(population[r1][k]-population[r2][k])
		else:
			temp=population[i][k]-0.5*(sin((2*pi*freq*g)+pi)*((g)/Gmax)+1)*(population[r1][k]-population[r2][k])
		result.append(temp)
	return result

#-----------------------------------------------eq 18---------------------------------------------
def claculate_mu_sigma(bita):
    gamma_1=gamma(bita+1)
    gamma_2=gamma((bita+1)/2)
    sin_var=pi*bita/2
    upper=gamma_1*sin(sin_var)
    pow_var=(bita-1)/2
    lower=gamma_2*bita*pow(2,pow_var)
    total_var=1/bita
    fract=upper/lower
    result=pow(fract,total_var)
    return result

def Levy(bita,mu,v):
    temp=abs(v)
    #print(temp)
    temp2=1/bita
    lower=pow(temp,temp2)
    if(lower==0):
    	lower+=0.0000001
    result=mu/lower
    return result

def normal_distribution(x, mean, sd):
    var = float(sd)**2
    denom = (2*pi*var)**.5
    num = exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

#X[i][d]ion=X[i][d]+(alpa(cross)Levy(bita))[d].(X[i][d]-X[best][d])
def Levy_distribution_strategy(i,d,alpha,bita,population):
    sigma_v=1
    sigma_mu=claculate_mu_sigma(bita)
    mu=norm(0, sigma_mu).cdf(population[i][d])
    v=norm(0, sigma_v).cdf(population[i][d])
    #print(v)
    result=population[i][d]+(alpha*Levy(bita,mu,v)*(population[i][d]-population[0][d]))
    return result

#-------------------------------------------------eq[21]-----------------------------------------

#X[i][d]Ion=X[i][d]Fi+(alpa(cross)Levy(bita))[d]*(ubd-lbd)

def equ_no_21(i,d,alpha,bita,popuation,ubd,lbd):
    sigma_v=1
    sigma_mu=claculate_mu_sigma(bita)
    mu=normal_distribution(population[i][d],0,sigma_mu)
    v=normal_distribution(population[i][d],0,sigma_v)
    x=alpha*Levy(bita,mu,v)
    temp=popuation[i][d]+(x*(ubd-lbd))
    return temp;
    
#------------------------------------------------eq 22-----------------------------------------

#X[i]Fu = X[i]Ion+alpha(cross)Levy(bita)(cross)(X[j]Ion-X[best]ion)

def equ_no_22(i,alpha,bita,popuation):
    result=list()
    sigma_v=1
    sigma_mu=claculate_mu_sigma(bita)
    for k in range(dim):
        mu=normal_distribution(population[i][k],0,sigma_mu)
        v=normal_distribution(population[i][k],0,sigma_v)
        x=alpha*Levy(bita,mu,v)
        dif=popuation[i][k]-popuation[0][k];
        temp=popuation[i][k]+x*dif
        result.append(temp);
    return result;


#--------------------------------num_of_feature fun-----------------------------------------------------
def fitness(feature_map,dim):
	result=list()
	fitn=0
	for j in range(dim):
		if(feature_map[j]==1):
			fitn+=1
	return fitn

#---------------------------num_feture for 2d--------------------------------------------------------

def cal_fitx(feature_map,population_count,dim):
	fitX=list()
	for i in range(population_count):
		temp=sum(feature_map[i])
		fitX.append(temp)
	return fitX

#---------------------------------sort a population----------------------------------------------
def make_sort(tempory,fitX,population_count):
	temp=fitX;
	result=list()
	temp.sort()
	i=0
	while (i<population_count):
		key=temp[i]
		for j in range(population_count):
			if(fitX==temp):
				result.append(tempory[j])
		while(i<population_count and temp[i]==key):
			i+=1;
	return result;


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


def returnAccuracy(x_train_cur,x_test_cur,y_train,y_test):
	#training-testing
	accuracy=knn.KNN(x_train_cur,y_train,x_test_cur,y_test)
	return accuracy

def returnAccuracyList(count,x_train,x_test,y_train,y_test,feature_map,fitX):
	accuracy_list=list()
	for i in range(count):
		#print("FEATURE MAP :",i)
		if(fitX[i]!=0):
			x_train_cur,x_test_cur=extractFeatures(x_train,x_test,feature_map[i])
			accuracy=returnAccuracy(x_train_cur,x_test_cur,y_train,y_test)
		else:
			accuracy=0
		accuracy_list.append(accuracy)
	return accuracy_list

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


#------------------------------------save the result------------------------------------------------

def saveInCSV_mini(feature_id,accuracy,population_count):
	fname='../NRO_knn/'+folder+'/res.csv'
	dim=29
	with open(fname,mode='a+') as result_file:
		result_writer=csv.writer(result_file)
		l=list()
		if(feature_id!=-1):
			l.append(feature_id)
		l.append(accuracy)
		result_writer.writerow(l)

#saving the result
def saveInCSV(feature_id,population,accuracy_list):
	fname='../NRO_knn/'+folder+'/'+str(feature_id)+'.csv'
	for i in range(len(population)):
		with open(fname,mode='a+') as result_file:
			result_writer=csv.writer(result_file)
			l=list()
			l.append(population[i])
			l.append(accuracy_list[i])
			result_writer.writerow(l)
		fname='../NRO_knn/'+folder+'/average.csv'
		with open(fname,mode='a+') as result_file:
			result_writer=csv.writer(result_file)
			l=list()
			l.append(feature_id)
			l.append(np.mean(accuracy_list))
			result_writer.writerow(l)

#------------------------result between 0-1-------------------------------------------------------
def population_for_f(population,population_count,dim):
	temp=list()
	for i in range(population_count):
		x=list()
		for j in range(dim):
			x.append(sigmoid(population[i][j]))
		temp.append(x)
	return temp


#--------------------------------------------------------graph--------------------------------------

def weightedGA_plot_graph():
	x=[]
	y=[]
	fname='../NRO_knn/'+folder+'/res.csv'
	cnt=0
	with open(fname, 'r') as csvfile:
		plots= csv.reader(csvfile, delimiter=',')
		for row in plots:
			cnt+=1
			x.append(float(row[1]))
			y.append(float(row[0]))

	plt.plot(x, y,'b', label='accuracy',marker='o')
	plt.title('Number of features vs Accuracy')
	plt.xlabel('Accuracy')
	plt.ylabel('Number of features')
	plt.show()



#--------------------------------main fun---------------------------------------------------------

#initialize the value of lb,ub,population_count,Max_iter,g
Max_iter=50;
g=1;
population_count=10;
lbd=0;
ubd=1;
dim=15
freq=2;

population=list()
Pa=list()
column_names=list()
x_train=list()
y_train=list()
x_test=list()
y_test=list()
accuracy_list=list()

#reading training/testing datasets
#column_names,x_train,y_train,train_count=read.read('../Data/1/train.csv')
#column_names,x_test,y_test,test_count=read.read('../Data/1/test.csv')
#column_names,x,y,train_count=read.read('../Data/UCI_DATA-master/'+folder+'/'+folder+'.csv')
#x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.20, random_state=1)
dataset='../Data/UCI_DATA-master/'+folder+'/'+folder+'.csv'
#column_names,x,y,train_count=read.read('../Data/UCI_DATA-master/'+folder+'/'+folder+'.csv')
#x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.20, random_state=1)
df = pd.read_csv(dataset)
a, b = np.shape(df)
data = df.values[:,0:b-1]
label = df.values[:,b-1]
dim = data.shape[1]
        
cross = 5
test_size = (1/cross)
x_train,x_test,y_train,y_test = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
#iris = load_iris()
population=init_population(population_count,dim)
#print(population)
tempory=population;
print(len(x_train[0]))
feature_map=make_feature(population,population_count,dim)
fitx=cal_fitx(feature_map,population_count,dim)
#print(feature_map)

#----------------------------------------------------------------
#evaluate the fitness funtion
#------------------------------------------------------------------
#print("OLD_POPULATION:",np.asarray(population).shape)
accuracy_list=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map,fitx)
saveInCSV(g,population,accuracy_list)
#print(population[0][0])
#print(sigmoid(population[0][0]))

Ne=list()
Pc=list()
alpha=0.01
bita=1.5
Pfi=0.5
while(g<Max_iter):
	g=g+1;
	print("ITERATION :",g)
	for i in range(population_count):
		rand_p=random.uniform(0, 1)
		#print(rand_p,end=" ")
		if(rand_p>Pfi):
			if(rand_p>P_beta):
				tempory[i]=odd_fission_SF(i,population,dim,rand_p,g)
			else:
				tempory[i]=odd_fission_PF(i,population,dim,rand_p,g)
		else:
			tempory[i]=even_fission_no_product(i,population,dim,g)
	population_new=tempory
	#print(population_new)
	#calculate the fit X
	#========================================================================================================
	feature_map=make_feature(population_for_f(population_new,population_count,dim),population_count,dim)
	fitx=cal_fitx(feature_map,population_count,dim)
	#print(feature_map)
	#print("After Fission:",np.asarray(population).shape)
	accuracy_list_new=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map,fitx)
	
	population.extend(population_new)
	accuracy_list.extend(accuracy_list_new)

	accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))

	population=list(population_res[0:10])
	accuracy_list=list(accuracy_res[0:10])
	#print (np.asarray(population).shape,np.asarray(accuracy_list).shape)
	saveInCSV(g,population,accuracy_list)

	#=======================================================================================================


	#NFI phase
	#Ionization stage
	rand=random.uniform(0,1)
	Pa=calculate_Pa(population_count)
	#print(Pa)
	for i in range(population_count):
		if(rand>Pa[i]):
			for d in range(dim):
				if(population[0][d]==population[population_count-1][d]):
					tempory[i][d]=Levy_distribution_strategy(i,d,alpha,bita,population)
					tempory[i][d]=equ_no_21(i,d,alpha,bita,tempory,ubd,lbd)
				else:
					tempory[i][d]=ionation_stage1(i,d,population,rand)
		else:
			for d in range(dim):
				if(population[0][d]==population[population_count-1][d]):
					tempory[i]=equ_no_22(i,alpha,bita,population)
				else:
					tempory[i][d]=equ_no_13(i,d,rand,population,dim)
	population_new=tempory
	#calculate the fit X
	#========================================================================================================
	feature_map=make_feature(population_for_f(population_new,population_count,dim),population_count,dim)
	fitx=cal_fitx(feature_map,population_count,dim)
	#print("After Ionization:",np.asarray(population).shape)
	accuracy_list_new=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map,fitx)
	
	population.extend(population_new)
	accuracy_list.extend(accuracy_list_new)

	accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))

	population=list(population_res[0:10])
	accuracy_list=list(accuracy_res[0:10])
	feature_map=make_feature(population_for_f(population,population_count,dim),population_count,dim)
	fitx=cal_fitx(feature_map,population_count,dim)
	number_fe=fitx
	#print (np.asarray(population).shape,np.asarray(accuracy_list).shape)
	saveInCSV(g,population,accuracy_list)

	#=======================================================================================================
	#Fusion stage
	Pc=calculate_Pc(population_count)
	rand_p=random.uniform(0,1)
	for i in range(population_count):
		if(rand_p<=Pc[i]):
			if(population[0][d]==population[population_count-1][d]):
				tempory[i]=equ_no_22(i,alpha,bita,population)
			else:
				tempory[i]=fusion_stage2(i,population,freq,rand_p,dim,g,Max_iter)
		else:
			tempory[i]=fusion_stage1(i,population,rand_p,dim)
	population_new=tempory
	#calculate the fitness function

	#===========================================================================================================
	feature_map=make_feature(population_for_f(population_new,population_count,dim),population_count,dim)
	fitx=cal_fitx(feature_map,population_count,dim)
	#print("NEW_POPULATION:",np.asarray(population).shape)
	accuracy_list_new=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map,fitx)
	
	population.extend(population_new)
	accuracy_list.extend(accuracy_list_new)
	number_fe.extend(fitx)
	number_fe,accuracy_res,population_res=zip(*sorted(zip(number_fe,accuracy_list,population)))

	accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))

	population=list(population_res[0:10])
	accuracy_list=list(accuracy_res[0:10])
	#print (np.asarray(population).shape,np.asarray(accuracy_list).shape)
	feature_map=make_feature(population_for_f(population,population_count,dim),population_count,dim)
	fitx=cal_fitx(feature_map,population_count,dim)
	saveInCSV(g,population,accuracy_list)
	saveInCSV_mini(np.mean(fitx),np.mean(accuracy_list),population_count)
print("Accuracy :",accuracy_list[0])
weightedGA_plot_graph()
	#============================================================================================================



