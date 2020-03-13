import random
from math import pi,exp,log

'''
    A variable rand_p determines reaction
'''
P_fi = 0.5 #probabiltiy of fission reaction
P_beta = 0.5 #probability of beta decay


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
        temp=(population[i][k]+population[j[k]])/2
        neutron.append(temp)
    return neutron

#theta calculation
#balance factor for exploitation and exploration
def theta_calculation(population,i,dim):
    population_count=len(population)
    theta=list()
    mag=0
    for j in range(dim):
        temp=(population[i][j]-population[0][j])**2
        mag=mag+temp
    mag=mag**(0.5)
    for j in range(dim):
        res=0
        for k in range(population_count):
            res=res+population[k][i]
        res=log(res)/res
        res=res*mag
        theta.append(res)
    return theta

def Gaussian_1D(mean,theta,x):
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


def odd_fission_SF(i,population,dim,rand_p):
    Nei=neutron_generation(i,i+1,population,dim)
    Pne=round(rand_p+1)
    theta=theta_calculation(population,i,dim)
    gaussian=Gaussian(population[0],theta,population[i],dim)
    result=list()
    for j in range(dim):
        temp=gaussian[j]+rand_p*population[0][j]-Pne*Nei[j]
        result.append(temp)
    return result

def odd_fission_PF(i,population,dim,rand_p):
    Nei=neutron_generation(i,i+1,population,dim)
    Pne=round(rand_p+1)
    r=random.randint(0,len(population)-1)
    while r==i:
        r=random.randint(0,len(population)-1)
    theta=theta_calculation(population,r,dim)
    gaussian=Gaussian(population[0],theta,population[i],dim)
    result=list()
    for j in range(dim):
        temp=gaussian[j]+rand_p*population[0][j]-Pne*Nei[j]
        result.append(temp)
    return result
    
def even_fission_no_product(i,population,dim):
    theta=theta_calculation(population,i,dim)
    gaussian=Gaussian(population[0],theta,population[i],dim)
    result=list()
    for j in range(dim):
        temp=gaussian[j]
        result.append(temp)
    return result


#------------------------------------------------
# #Nuclear Fussion Phase (NFu)
#------------------------------------------------
# odd nuclei can generate 2 products SF or PF


#------------------------------------------eq 14-----------------------------------------
#pc[i]=rank(fitX[i]Ion)/N

def calculate_Pc(j,population,N,dim,fitX):#how to claculate fitX
    Pc=list()
    for k in range(dim):
        rank_of_jth_pop=rank(fitX[j][k]);#confusion how to claculate the rank function and fitof a popuation is not calculated
        temp=rank_of_jth_pop/N
        Pc.append(temp)
    return Pc;

#---------------------------------------------eq 15----------------------------------------
#X[j]=X[j]Ion+rand*(X[r1]IOn-X[best]Ion)-e(-norm(Xr1Ion-Xr2Ion)).(Xr1Ion-Xr2Ion)
from math import exp
from cmath import phase
def fusion_stage1(j,population,r1,r2,rand_p,dim):
    result=list()
    for i in range(dim):
        x=phase(population[r1][i],0-population[r2][i])
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
    retrun res

#X[j]Fu=X[j]Ion-0.5(sin(2*pi*freq*g+pi).Gmax-g/Gmax +1)(X[r1]ion-X[r1]ion)...if rand>0.5
#X[j]Fu=X[j]Ion-0.5(sin(2*pi*freq*g+pi).g/Gmax +1)(X[r1]ion-X[r1]ion)...if rand<=0.5
from math import sin
def fusion_stage2(i,population,r1,r2,freq,rand_p):
    reslut=list()
    for k in range(dim):
        g=claculate_g(population,k)
        if(rand_p>0.5):
            result=population[i][k]-0.5(sin((2*pi*freq*g)+pi)*((Gmax-g)/Gmax)+1)*(population[r1][k]-population[r2][k])  #confusion what is Gmax
        else:
            result=population[i][k]-0.5(sin((2*pi*freq*g)+pi)*((g)/Gmax)+1)*(population[r1][k]-population[r2][k])



    

