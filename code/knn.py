from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def KNN(x_train,y_train,x_test,y_test):
	clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(x_train,y_train)
	accuracy=clf.score(x_test,y_test) 
	print ("Accuracy: ",accuracy)
	return accuracy
  