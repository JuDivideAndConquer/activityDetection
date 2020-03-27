import random
from math import pi,exp,log,pow,gamma
from cmath import phase

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
# #Ionization stage
#------------------------------------------------

#------------------------------------------eq 10-----------------------------------------
#pa[i]=rank(fitX[i]Fi)/N

def calculate_Pa(j,population,N,dim,fitX):#how to claculate fitX
    Pa=list()
    for k in range(dim):
        rank_of_jth_pop=rank(fitX[j][k]);#confusion how to claculate the rank function and fitof a popuation is not calculated
        temp=rank_of_jth_pop/N
        Pa.append(temp)
    return Pa;

#------------------------------------------------------eq11 &eq 12-------------------------

#X[i][d]ion=X[r1][d]Fi-rand*(X[r2][d]Fi-X[i][d]Fi)...if rand<=0.5
#X[i][d]ion=X[r1][d]Fi-rand*(X[r2][d]Fi-X[i][d]Fi)...if rand>0.5
def ionation_stage1(i,d,r1,r2,popuation,rand):
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
        rank_of_jth_pop=rank(fitX[j][k]);#confusion how to claculate the rank function and fitof a popuation is not calculated
        temp=rank_of_jth_pop/N
        Pc.append(temp)
    return Pc;

#---------------------------------------------eq 15----------------------------------------
#X[j]=X[j]Ion+rand*(X[r1]IOn-X[best]Ion)-e(-norm(Xr1Ion-Xr2Ion)).(Xr1Ion-Xr2Ion)
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
    return res

#X[j]Fu=X[j]Ion-0.5(sin(2*pi*freq*g+pi).Gmax-g/Gmax +1)(X[r1]ion-X[r1]ion)...if rand>0.5
#X[j]Fu=X[j]Ion-0.5(sin(2*pi*freq*g+pi).g/Gmax +1)(X[r1]ion-X[r1]ion)...if rand<=0.5
def fusion_stage2(i,population,r1,r2,freq,rand_p,dim):
    reslut=list()
    Gmax=100
    for k in range(dim):
        g=claculate_g(population,k)
        if(rand_p>0.5):
            temp=population[i][k]-0.5(sin((2*pi*freq*g)+pi)*((Gmax-g)/Gmax)+1)*(population[r1][k]-population[r2][k])
        else:
            temp=population[i][k]-0.5(sin((2*pi*freq*g)+pi)*((g)/Gmax)+1)*(population[r1][k]-population[r2][k])
        result.append(temp)
    return result

#-----------------------------------------------eq 18---------------------------------------------
def calculate_mu_sigma(bita):
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
    temp2=1/bita
    lower=pow(temp,temp2)
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
    result=population[i][d]+(Xor(alpha,Levy(beta,mu,v))*(population[i][d]-population[0][d]))
    return result

#-------------------------------------------------eq[21]-----------------------------------------

#X[i][d]Ion=X[i][d]Fi+(alpa(cross)Levy(bita))[d]*(ubd-lbd)

def equ_no_22(i,d,alpha,bita,popuation,ubd,lbd):
    sigma_v=1
    sigma_mu=claculate_mu_sigma(bita)
    mu=normal_distribution(population[i][k],0,sigma_mu)
    v=normal_distribution(population[i][k],0,sigma_v)
    x=Xor(alpha,Levy(bita,mu,v))
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
        x=Xor(alpha,Levy(bita,mu,v))
        dif=popuation[i][k]-popuation[0][k];
        temp=popuation[i][k]+(Xor(x,dif))
        result.append(temp);
    return result;
