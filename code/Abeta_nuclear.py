
import random
from math import pi,log,pow,gamma,sin,sqrt,exp
from cmath import phase
import svm
import read
import numpy as np
from sklearn.model_selection import train_test_split
import importlib
import csv
from scipy.stats import norm
import sailFish
'''
    A variable rand_p determines reaction
'''
P_fi = 0.5 #probabiltiy of fission reaction
P_beta = 0.75 #probability of beta decay
pp=0.1 #parameter
ID_POS = 0
ID_FIT = 1
omega = 0.9
dim=59;



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

#------------------------------------------------
#sailfishcode
#------------------------------------------------

#--------------------------------update selfish-------------------------------------------------
def sailFish(trainX, testX, trainy, testy ,dimension,pop_size):
		#url = "https://raw.githubusercontent.com/Rangerix/UCI_DATA/master/CSVformat/BreastCancer.csv"
		#df = pd.read_csv(dataset)
		#a, b = np.shape(df)
		#data = df.values[:,0:b-1]
		#label = df.values[:,b-1]
		#dimension = data.shape[1]

		cross = 5
		test_size = (1/cross)
		#trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
		clf=KNeighborsClassifier(n_neighbors=5)
		clf.fit(trainX,trainy)
		val=clf.score(testX,testy)
		whole_accuracy = val
		print("Total Acc: ",val)

		s_size = int(pop_size / pp)
		sf_pop = initialise(pop_size, dimension, trainX, testX, trainy, testy)
		s_pop = initialise(s_size, dimension, trainX, testX, trainy, testy)

		sf_gbest = _get_global_best__(sf_pop, ID_FIT, ID_MIN_PROBLEM)
		s_gbest = _get_global_best__(s_pop, ID_FIT, ID_MIN_PROBLEM)

		temp = np.array([])

		for iterno in range(0, epoch):
			print(iterno)
			## Calculate lamda_i using Eq.(7)
			## Update the position of sailfish using Eq.(6)
			for i in range(0, pop_size):
				PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
				lamda_i = 2 * np.random.uniform() * PD - PD
				sf_pop_arr = s_gbest[ID_POS] - lamda_i * ( np.random.uniform() *( sf_gbest[ID_POS] + s_gbest[ID_POS] ) / 2 - sf_pop[i][ID_POS] )
				sf_pop_fit = sf_pop[i][ID_FIT]
				new_tuple = (sf_pop_arr, sf_pop_fit)

				sf_pop[i] = new_tuple

			## Calculate AttackPower using Eq.(10)
			AP = A * ( 1 - 2 * (iterno) * epxilon )
			if AP < 0.5:
				alpha = int(len(s_pop) * AP )
				beta = int(dimension * AP)
				### Random choice number of sardines which will be updated their position
				list1 = np.random.choice(range(0, len(s_pop)), alpha)
				for i in range(0, len(s_pop)):
					if i in list1:
						#### Random choice number of dimensions in sardines updated
						list2 = np.random.choice(range(0, dimension), beta)
						s_pop_arr = s_pop[i][ID_POS]
						for j in range(0, dimension):
							if j in list2:
								##### Update the position of selected sardines and selected their dimensions
								s_pop_arr[j] = np.random.uniform()*( sf_gbest[ID_POS][j] - s_pop[i][ID_POS][j] + AP )
						s_pop_fit = s_pop[i][ID_FIT]
						new_tuple = ( s_pop_arr, s_pop_fit)
						s_pop[i] = new_tuple
			else:
				### Update the position of all sardine using Eq.(9)
				for i in range(0, len(s_pop)):
					s_pop_arr = np.random.uniform()*( sf_gbest[ID_POS] - s_pop[i][ID_POS] + AP )
					s_pop_fit = s_pop[i][ID_FIT]
					new_tuple = (s_pop_arr, s_pop_fit)
					s_pop[i] = new_tuple

			# population in binary
			# y, z = np.array([]), np.array([])
			# ychosen = 0
			# zchosen = 0
			# # print(np.shape(s_pop))
			for i in range(np.shape(s_pop)[0]):
				agent = s_pop[i][ID_POS]
				tempFit = s_pop[i][ID_FIT]
				random.seed(time.time())
				#print("agent shape :",np.shape(agent))
				y, z = np.array([]), np.array([])
				for j in range(np.shape(agent)[0]):
					random.seed(time.time()*200+999)
					r1 = random.random()
					random.seed(time.time()*200+999)
					if sigmoid1(agent[j]) < r1:
						y = np.append(y,0)
					else:
						y = np.append(y,1)

				yfit = fitness(y, trainX, testX, trainy, testy)
				agent = deepcopy(y)
				tempFit = yfit

				new_tuple = (agent,tempFit)
				s_pop[i] = new_tuple
			## Recalculate the fitness of all sardine
			# print("y chosen:",ychosen,"z chosen:",zchosen,"total: ",ychosen+zchosen)
			for i in range(0, len(s_pop)):
				s_pop_arr = s_pop[i][ID_POS]
				s_pop_fit = fitness(s_pop[i][ID_POS],trainX, testX, trainy, testy)
				new_tuple = (s_pop_arr, s_pop_fit)
				s_pop[i] = new_tuple

			# local search algo
			for i in range(np.shape(s_pop)[0]):
				new_tuple = adaptiveBeta(s_pop[i],trainX,testX,trainy,testy)
				s_pop[i] = new_tuple

			## Sort the population of sailfish and sardine (for reducing computational cost)
			sf_pop = sorted(sf_pop, key=lambda temp: temp[ID_FIT])
			s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT])
			for i in range(0, pop_size):
				s_size_2 = len(s_pop)
				if s_size_2 == 0:
					s_pop = initialise(s_pop, dimension, trainX, testX, trainy, testy)
					s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT])
				for j in range(0, s_size):
					### If there is a better solution in sardine population.
					if sf_pop[i][ID_FIT] > s_pop[j][ID_FIT]:
						sf_pop[i] = deepcopy(s_pop[j])
						del s_pop[j]
					break   #### This simple keyword helped reducing ton of comparing operation.
							#### Especially when sardine pop size >> sailfish pop size
			
			# OBL
			# sf_pop = OBL(sf_pop, trainX, testX, trainy, testy)
			sf_current_best = _get_global_best__(sf_pop, ID_FIT, ID_MIN_PROBLEM)
			s_current_best = _get_global_best__(s_pop, ID_FIT, ID_MIN_PROBLEM)
			if sf_current_best[ID_FIT] < sf_gbest[ID_FIT]:
				sf_gbest = np.array(deepcopy(sf_current_best))
			if s_current_best[ID_FIT] < s_gbest[ID_FIT]:
				s_gbest = np.array(deepcopy(s_current_best))


		testAcc = test_accuracy(sf_gbest[ID_POS], trainX, testX, trainy, testy)
		featCnt = onecnt(sf_gbest[ID_POS])
		print("Test Accuracy: ", testAcc)
		print("#Features: ", featCnt)

		return sf_gbest[ID_POS], testAcc, featCnt

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
		sf,fetures,accuracy=sailFish(x_train_cur,x_test_cur,y_train,y_test,dim,population_count)
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

#saving the result
def saveInCSV(feature_id,population,accuracy_list):
	fname='../Result/Sonar1/'+str(feature_id)+'.csv'
	for i in range(len(population)):
		with open(fname,mode='a+') as result_file:
			result_writer=csv.writer(result_file)
			l=list()
			l.append(population[i])
			l.append(accuracy_list[i])
			result_writer.writerow(l)
		fname='../Result/Sonar1/average.csv'
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


#--------------------------------main fun---------------------------------------------------------

#initialize the value of lb,ub,population_count,Max_iter,g
Max_iter=50;
g=1;
population_count=10;
lbd=0;
ubd=1;
freq=2;

population=list()
Pa=list()
fitX=list()
population=init_population(population_count,dim)
#print(population)
tempory=population;
column_names=list()
x_train=list()
y_train=list()
x_test=list()
y_test=list()
accuracy_list=list()

#reading training/testing datasets
#column_names,x_train,y_train,train_count=read.read('../Data/1/train.csv')
#column_names,x_test,y_test,test_count=read.read('../Data/1/test.csv')
column_names,x,y,train_count=read.read('../Data/UCI_DATA-master/BreastEW/BreastEW.csv')
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.20, random_state=1)
print(len(x_train[0]))
feature_map=make_feature(population,population_count,dim)
#print(feature_map)

#----------------------------------------------------------------
#evaluate the fitness funtion
#------------------------------------------------------------------
print("OLD_POPULATION:",np.asarray(population).shape)
accuracy_list=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map)
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
	#print(feature_map)
	print("After Fission:",np.asarray(population).shape)
	accuracy_list_new=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map)
	
	population.extend(population_new)
	accuracy_list.extend(accuracy_list_new)

	accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))

	population=list(population_res[0:10])
	accuracy_list=list(accuracy_res[0:10])
	print (np.asarray(population).shape,np.asarray(accuracy_list).shape)
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
	print("After Ionization:",np.asarray(population).shape)
	accuracy_list_new=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map)
	
	population.extend(population_new)
	accuracy_list.extend(accuracy_list_new)

	accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))

	population=list(population_res[0:10])
	accuracy_list=list(accuracy_res[0:10])
	print (np.asarray(population).shape,np.asarray(accuracy_list).shape)
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
	print("NEW_POPULATION:",np.asarray(population).shape)
	accuracy_list_new=returnAccuracyList(population_count,x_train,x_test,y_train,y_test,feature_map)
	
	population.extend(population_new)
	accuracy_list.extend(accuracy_list_new)

	accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population),reverse=True))

	population=list(population_res[0:10])
	accuracy_list=list(accuracy_res[0:10])
	print (np.asarray(population).shape,np.asarray(accuracy_list).shape)
	saveInCSV(g,population,accuracy_list)

	#============================================================================================================
	s_size = int(population_count/ pp)


