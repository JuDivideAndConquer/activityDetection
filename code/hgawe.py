#Hybrid genetic Algorithm with Wrapper Embedded Feature Selection approach
import random
from sklearn.linear_model import RidgeClassifier
import read
from sklearn.model_selection import train_test_split
import save_in_csv
import numpy as np

#INITIALIZATION
population_size=20
crossover_probability=0.85
mutation_probability=0.8
stopping_point=15
chromozome_size=0
alpha_min=15.0
alpha_max=16.0
x_train=list()
y_train=list()
x_test=list()
y_test=list()


def init_system():
    global x_train,x_test,y_train,y_test,chromozome_size
    column_names,data,label,temp=read.read("../Data/2/UCI_DATA/CSVformat/BreastCancer.csv")
    x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.33,random_state=15)
    chromozome_size=len(x_test[0])

def extractFeatures(feature_map):
        x_train_cur=list()
        x_test_cur=list()
        index_list = [i for i, val in enumerate(feature_map) if val]
        #making new train set with new features
        for j in range(len(x_train)):
                res_list = [x_train[j][i] for i in index_list]
                x_train_cur.append(res_list)
         #making new test set with new features       
        for j in range(len(x_test)):
            res_list = [x_test[j][i] for i in index_list]
            x_test_cur.append(res_list)
        return x_train_cur,x_test_cur




def fitness_evaluation(feature_map):
    print(feature_map)
    intron=feature_map[0]
    extron=list(feature_map[1:])
    x_train_cur,x_test_cur=extractFeatures(extron)
    x_train_cur=np.asarray(x_train_cur).astype(np.float64)
    x_test_cur=np.asarray(x_test_cur).astype(np.float64)
    clf=RidgeClassifier(alpha=intron).fit(x_train_cur,y_train)
    score=clf.score(x_test_cur,y_test)
    return score



def init_population():
    population=list()
    for i in range(population_size):
        temp=list()
        extron=list()
        #extron generation
        for j in range(chromozome_size):
            x=random.randint(0,1)
            extron.append(x)
        #intron generation
        alpha=alpha_min
        while(alpha<alpha_max):
            feature_map=list()
            feature_map.append(alpha)
            feature_map.extend(extron)
            score0=fitness_evaluation(feature_map)
            alpha+=0.5
            feature_map=list()
            feature_map.append(alpha)
            feature_map.extend(extron)
            score1=fitness_evaluation(feature_map)
            if score1<=score0:
                alpha-=0.5
                break
        temp.append(alpha)
        temp.extend(extron)
        population.append(temp)
    return population




def fitness_evaluation_for_population(population):
    score_list=list()
    for i in range(population_size):
        score=fitness_evaluation(population[i])
        score_list.append(score)
    return score_list


def flip(x):
    if x==0:
        return 1
    else:
        return 0

def check_if_zero_features(x):
    t=0
    for i in range(len(x)):
        if x[i]==1:
            t+=1
    if t==0:
        return 1
    else :
        return 0


def offspring_generation(parent1,parent2):
    intron1=parent1[0]
    extron1=list(parent1[1:])
    intron2=parent2[0]
    extron2=list(parent2[1:])
    child1=list()
    child2=list()
    child1.append(intron1)
    child2.append(intron2)  
    crossover_point=random.randint(0,chromozome_size)
    child1.extend(extron1[0:crossover_point])
    child1.extend(extron2[crossover_point:])
    child2.extend(extron2[0:crossover_point])
    child2.extend(extron1[crossover_point:])
    for i in range(chromozome_size):
        p_extron1=random.random()
        p_extron2=random.random()
        if(p_extron1>mutation_probability):
            child1[i]=flip(child1[i])
        if(p_extron2>mutation_probability):
            child2[i]=flip(child2[i])
    while check_if_zero_features(child1) or check_if_zero_features(child2):
        child1,child2=offspring_generation(parent1,parent2)
    return child1,child2

def generate_new_population(population):
    population_new=list()
    for i in range(population_size):
        parent1=population[i]
        if(i==population_size-1):
            parent2=population[0]
        else:
            parent2=population[i+1]
        child1,child2=offspring_generation(parent1,parent2)
        population_new.append(child1)
        population_new.append(child2)
    return population_new

def ga(population):
    iteration=0
    score_list=fitness_evaluation_for_population(population)
    while iteration<stopping_point:
        print("\titeration_inside_ga :",iteration)
        population_new=generate_new_population(population)
        score_list_new=fitness_evaluation_for_population(population_new)
        population.extend(population_new)
        score_list.extend(score_list_new)
        score_res,population_res=zip(*sorted(zip(score_list,population),reverse=True))
        population=list(population_res[0:population_size])
        score_list=list(score_res[0:population_size])
        #save_in_csv.saveInCSV(iteration,population,score_list,'../Result/HGAWE/uci_breastcancer/')
        iteration+=1
    return population

def offspring_generation_hybrid(parent1,parent2):
    intron1=parent1[0]
    extron1=list(parent1[1:])
    intron2=parent2[0]
    extron2=list(parent2[1:])
    child1=list()
    child2=list()

    p_intron=random.random()
    if p_intron>crossover_probability:
        child1.append(intron2)
        child2.append(intron1)
    else:
        child1.append(intron1)
        child2.append(intron2)  
    child1.extend(extron1)
    child2.extend(extron2)
    return child1,child2

def generate_new_population_hybrid(population):
    population_new=list()
    for i in range(population_size):
        parent1=population[i]
        if(i==population_size-1):
            parent2=population[0]
        else:
            parent2=population[i+1]
        child1,child2=offspring_generation_hybrid(parent1,parent2)
        population_new.append(child1)
        population_new.append(child2)
    return population_new

def hgawe():
    iteration=0
    init_system()
    population=init_population()
    score_list=fitness_evaluation_for_population(population)
    while iteration<stopping_point:
        print("ITERATION :",iteration)
        population_new=ga(population)
        population_new=generate_new_population_hybrid(population_new)
        score_list_new=fitness_evaluation_for_population(population_new)
        population.extend(population_new)
        score_list.extend(score_list_new)
        score_res,population_res=zip(*sorted(zip(score_list,population),reverse=True))
        population=list(population_res[0:population_size])
        score_list=list(score_res[0:population_size])
        save_in_csv.saveInCSV(iteration,population,score_list,'../Result/HGAWE/uci_breastcancer/')
        iteration+=1

hgawe()