'''
tot_fts -> count of total features
cur_population -> int[cur_count][tot_fts]
cur_acc -> float[cur_count]
cur_count -> total entities in current population
'''
import importlib
import csv
import random
import svm
import read
import numpy as np 


#implementing genetic algo
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

#initailaze population randomly
def init_population(count=20,tot_fts=561):
        init_popul=list()
        j=0
        while(j<count):
                new=list()
                for i in range(tot_fts):
                        bit=random.randint(0,1)
                        new.append(bit)
                if (new not in init_popul):
                        init_popul.append(new)
                        j+=1
                
        return init_popul


#generating new population
def geneticAlgo(tot_fts_count,cur_population,population_count):
        new_population=list()
        new_count=0
        i=0
        while i<population_count:
                #finding cross-over and mutation points
                crossOverPoint = CrossOver(tot_fts_count)
                mutationPoint = Mutation(tot_fts_count)

                #carrying it out
                child1=list()
                print(np.asarray(cur_population[i]).shape)
                child1.extend(cur_population[i][0:crossOverPoint])

                if(i==population_count-1):
                        child1.extend(cur_population[0][crossOverPoint:])
                else:
                        child1.extend(cur_population[i+1][crossOverPoint:])
                child1=flip(child1,mutationPoint)
                
                #adding to new population
                if child1 not in cur_population and child1 not in new_population:
                        new_population.append(child1)
                        new_count+=1
                        i+=1
        return new_population

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

#saving the result
def saveInCSV(feature_id,population,accuracy_list):
        fname='../Result/GA/'+str(feature_id)+'.csv'
        for i in range(len(population)):
                with open(fname,mode='a+') as result_file:
                        result_writer=csv.writer(result_file)
                        l=list()
                        l.append(population[i])
                        l.append(accuracy_list[i])
                        result_writer.writerow(l)
        fname='../Result/GA/average.csv'
        with open(fname,mode='a+') as result_file:
                result_writer=csv.writer(result_file)
                l=list()
                l.append(feature_id)
                l.append(np.mean(accuracy_list))
                result_writer.writerow(l)


#run genetic algo for <epochs>
def runGeneticAlgo(epochs,count=20,tot_fts_count=561):
        iteration=0
        column_names=list()
        x_train=list()
        y_train=list()
        x_test=list()
        y_test=list()
        accuracy_list=list()

        #reading training/testing datasets
        column_names,x_train,y_train,train_count=read.read('../Data/1/train.csv')
        column_names,x_test,y_test,test_count=read.read('../Data/1/test.csv')
        
        population_old=init_population()
        print("OLD_POPULATION:",np.asarray(population_old).shape)
        
        #population_new=geneticAlgo(tot_fts_count,population_old,count)
        #print("NEW_POPULATION:",np.asarray(population_new).shape)


        print("ITERATION :",iteration)
        accuracy_list=returnAccuracyList(count,x_train,x_test,y_train,y_test,population_old)
        #print("ACCURACY LIST:",np.asarray(accuracy_list).shape)
        saveInCSV(iteration,population_old,accuracy_list)

        iteration+=1
        

        while(iteration<epochs):
                print("ITERATION :",iteration)
                population_new=geneticAlgo(tot_fts_count,population_old,count)
                print("NEW_POPULATION:",np.asarray(population_new).shape)

                accuracy_list_new=returnAccuracyList(count,x_train,x_test,y_train,y_test,population_new)

                population_old.extend(population_new)
                accuracy_list.extend(accuracy_list_new)

                accuracy_res,population_res=zip(*sorted(zip(accuracy_list,population_old),reverse=True))

                population_old=list(population_res[0:20])
                accuracy_list=list(accuracy_res[0:20])
                print (np.asarray(population_old).shape,np.asarray(accuracy_list).shape)

                saveInCSV(iteration,population_old,accuracy_list)
                iteration+=1
                










    









