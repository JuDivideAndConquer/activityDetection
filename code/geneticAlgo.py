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
        return obj
  
def geneticAlgo(tot_fts,cur_population,cur_count):
        new_population=list()
        new_count=0
        for i in range(cur_count-1):
                #finding cross-over and mutation points
                crossOverPoint = CrossOver(tot_fts)
                mutationPoint = Mutation(tot_fts)

                #carrying it out
                child1=list()
                child1.extend(cur_population[i][0:crossOverPoint])
                child1.extend(cur_population[i+1][crossOverPoint:])
                child2=list()
                child2.extend(cur_population[i+1][0:crossOverPoint])
                child2.extend(cur_population[i][crossOverPoint:])
                child1=flip(child1,mutationPoint)
                child2=flip(child2,mutationPoint)

                #adding to new population
                if child1 not in cur_population and child1 not in new_population:
                        new_population.append(child1)
                        new_count+=1
                elif child2 not in cur_population and child2 not in new_population:
                        new_population.append(child2)
                        new_count+=1
        return new_population,cur_population


