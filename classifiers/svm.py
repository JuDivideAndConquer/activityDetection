from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.fixes import loguniform

def SVM(x_train,y_train,x_test,y_test):
    params={'C': loguniform(1e0, 1e3),'gamma': loguniform(1e-4, 1e-3),'kernel': ['rbf','linear'],'class_weight':['balanced', None]}
    svm= SVC()
    clf=GridSearchCV(svm,params)
    clf.fit(x_train, y_train) 
    svm_predictions = clf.predict(x_test) 
    accuracy = accuracy_score(y_test,svm_predictions)
    print ("Accuracy: ",accuracy)
    print(clf.get_params)
    return accuracy
  
