from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn(x_train,y_train,x_test,y_test):
	neigh=KNeighborsClassifier(n_neighbors=3)
	neigh.fit(x_train,y_train)
	knn_predictions = neigh.predict(x_test) 
	accuracy = accuracy_score(y_test,knn_predictions)
	print ("Accuracy: ",accuracy)
	return accuracy
  