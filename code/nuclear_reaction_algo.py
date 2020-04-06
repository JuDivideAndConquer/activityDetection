#problem a gaussina distribution
#problem in levy distribution
#problem in nutron generation
#problem in sort the population

import random
from math import pi,exp,log,pow,gamma,sin
from cmath import phase
import svm
import read
import numpy as np

'''
    A variable rand_p determines reaction
'''
P_fi = 0.5 #probabiltiy of fission reaction
P_beta = 0.5 #probability of beta decay


def Xor(a,b):
	return (a^b)

#initializing the population
def init_population(population_count,dim):
    population=list()
    for i in range(population_count):
        x=list()
        for j in range(dim):
            x.append(0+random.randrange(0,1)*1)
        population.append(x)
    return population

#Heated neutron generation
# Nei <- (xi+xj)/2
#xi,xj are nuclei
def neutron_generation(i,j,population,dim):
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
		return mean
	exponent=-0.5*((x-mean)/theta)**2
	coefficient=(theta*((2*pi)**(0.5)))**(-1)
	result=coefficient*exp(exponent)
	return result

def Gaussian(mean,theta,x,dim): #dim dimension
    result=list()
    for i in range(dim):
        temp=Gaussian_1D(mean[i],theta[i],x[i])
        result.append(temp)
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
		temp=gaussian[j]+rand_p*population[0][j]-Pne*Nei[j]
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
    return result

#------------------------------------------------
# #Ionization stage
#------------------------------------------------

#------------------------------------------eq 10-----------------------------------------
#pa[i]=rank(fitX[i]Fi)/N

def calculate_Pa(j,population,N,dim,fitX):#how to claculate fitX
    Pa=list()
    for k in range(dim):
        rank_of_jth_pop=j#confusion how to claculate the rank function and fitof a popuation is not calculated
        temp=rank_of_jth_pop/N
        Pa.append(temp)
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


#------------------------------------------------
# #Nuclear Fussion Phase (NFu)
#------------------------------------------------
# odd nuclei can generate 2 products SF or PF


#------------------------------------------eq 14-----------------------------------------
#pc[i]=rank(fitX[i]Ion)/N

def calculate_Pc(j,population,N,dim,fitX):#how to claculate fitX
    Pc=list()
    for k in range(dim):
        rank_of_jth_pop=j;#confusion how to claculate the rank function and fitof a popuation is not calculated
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
def fusion_stage2(i,population,freq,rand_p,dim,g):
	r1=random.randint(0,population_count-1)
	r2=random.randint(0,population_count-1)
	result=list()
	Gmax=15
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
    	return lower
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
    mu=normal_distribution(population[i][d],0,sigma_mu)
    v=normal_distribution(population[i][d],0,sigma_v)
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


#--------------------------------fitness fun-----------------------------------------------------
def fitness(population,population_count,dim):
	result=list()
	for i in range(population_count):
		fitn=0
		for j in range(dim):
			if(population[i][j]>=0.5):
				fitn+=1
		result.append(fitn)
	return result

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


#--------------------------------main fun---------------------------------------------------------

#initialize the value of lb,ub,population_count,Max_iter,g
Max_iter=15;
g=0;
population_count=516;
lbd=0;
ubd=1;
dim=20;
freq=2;

population=list()
Pa=list()
fitX=list()
population=init_population(population_count,dim)
tempory=population;

#reading training/testing datasets
column_names,x_train,y_train,train_count=read.read('../Data/1/train.csv')
column_names,x_test,y_test,test_count=read.read('../Data/1/test.csv')
print("OLD_POPULATION:",np.asarray(population).shape)
accuracy_list=returnAccuracyList(dim,x_train,x_test,y_train,y_test,population)

#----------------------------------------------------------------
#evaluate the fitness funtion
#------------------------------------------------------------------

fitX=fitness(population,population_count,dim)
Ne=list()
Pc=list()
alpha=0.01
bita=1.5
while(g<Max_iter):
	g=g+1;
	print("ITERATION :",g)
	for i in range(population_count):
		rand_p=random.randrange(0,1)
		tempory[i]=odd_fission_SF(i,population,dim,rand_p,g)
		tempory[i]=odd_fission_PF(i,population,dim,rand_p,g)
		tempory[i]=even_fission_no_product(i,population,dim,g)
		fitX=fitness(tempory,population_count,dim)
		population=make_sort(tempory,fitX,population_count)
	#NFI phase
	#Ionization stage
	for i in range(population_count):
		Pa.append(calculate_Pa(i,population,population_count,dim,fitX))#how to claculate FitX
	for i in range(population_count):
		for d in range(dim):
			rand=random.random()
			population[i][d]=Levy_distribution_strategy(i,d,alpha,bita,population)
			population[i][d]=equ_no_21(i,d,alpha,bita,population,ubd,lbd)
			population[i]=equ_no_22(i,alpha,bita,population)
			population[i][d]=ionation_stage1(i,d,population,rand);
			#popuation[i][d]=#where is equation n0 13 ??
		#calculate the fit X
		fitX=fitness(population,population_count,dim)
	#Fusion stage
	for j in range(population_count):
		Pc.append(calculate_Pc(j,population,population_count,dim,fitX))
	for i in range(population_count):
		population[i]=equ_no_22(i,alpha,bita,population)
		population[i]=fusion_stage1(i,population,rand_p,dim)
		population[i]=fusion_stage2(i,population,freq,rand_p,dim,g)
		#calculate the fitness function
		fitX=fitness(population,population_count,dim)
	print("NEW_POPULATION:",np.asarray(population).shape)









