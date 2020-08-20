from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def KNN(x_train,y_train,x_test,y_test):
	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(x_train,y_train)
	accuracy=accuracy_score(y_test,clf.predict(x_test)) 
	#print ("Accuracy: ",accuracy)
	return accuracy
  
