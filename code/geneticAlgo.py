'''
tot_fts -> count of total features
cur_population -> int[cur_count][tot_fts]
cur_acc -> float[cur_count]
cur_count -> total entities in current population
'''

import random

def CrossOver(length):
        pos=random.randint(0,length-1)
        return pos

def Mutation(length):
        pos=random.randint(0,length-1)
        return pos

def  flip(obj,pos):
        if( obj[pos]==0):
                obj[pos]=1
        else:
                obj[pos]=0
  
def geneticAlgo(tot_fts,cur_population,cur_count):
        new_population=list()
        new_count=0
        for i in range(cur_count-1):
                #finding cross-over and mutation points
                crossOverPoint = CrossOver(tot_fts)
                mutationPoint = Mutation(tot_fts)

                #carrying it out
                child1=cur_population[i][0:crossOverPoint].extend(cur_population[i+1][crossOverPoint:])
                child2=cur_population[i+1][0:crossOverPoint].extend(cur_population[i][crossOverPoint:])
                child1=flip(child1,mutationPoint)
                child2=flip(child2,mutationPoint)

                #adding to new population
                new_population.add(child1)
                new_population.add(child2)
                new_count=new_count+2


