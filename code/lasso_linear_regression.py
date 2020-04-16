from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import accuracy_score

def lasso(x_train,y_train,x_test,y_test):
    reg=ElasticNetCV(cv=5,random_state=0).fit(x_train,y_train)
    for i in range(len(x_test)):
        for j in range(len(x_test[0])):
            x_test[i][j]=float(x_test[i][j])
    lasso_predictions = reg.predict(x_test) 
    accuracy = accuracy_score(y_test,lasso_predictions)
    print ("Accuracy: ",accuracy)
    #print(clf.get_params)
    return accuracy