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
    theta=theta_calculation(population,i,dim)
    gaussian=Gaussian(population[0],theta,population[i],dim)
    result=list()
    for j in range(dim):
        temp=gaussian[j]+rand_p*population[0][j]-Pne*Nei[j]
        result.append(temp)
    return result
    

    

