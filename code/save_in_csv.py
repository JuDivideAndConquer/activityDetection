import numpy as np
import csv

def saveInCSV_mini(feature_map,accuracy,fname,feature_id=-1):
        with open(fname,mode='a+') as result_file:
                result_writer=csv.writer(result_file)
                l=list()
                if(feature_id!=-1):
                    l.append(feature_id)    
                l.append(feature_map)
                l.append(accuracy)
                result_writer.writerow(l)

#saving the result
def saveInCSV(feature_id,population,accuracy_list,path):
        fname=path+str(feature_id)+'.csv'
        for i in range(len(population)):
                saveInCSV_mini(population[i],accuracy_list[i],fname)
        fname=path+'average.csv'
        with open(fname,mode='a+') as result_file:
                result_writer=csv.writer(result_file)
                l=list()
                l.append(feature_id)
                l.append(np.mean(accuracy_list))
                result_writer.writerow(l)