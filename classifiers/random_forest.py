# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
def randomforest(x_train,y_train,x_test,y_test):
	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.transform(y_test)
	regressor = RandomForestRegressor(n_estimators=20, random_state=0)
	regressor.fit(x_train, y_train)
	y_pred = regressor.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(confusion_matrix(y_test,y_pred))
	print(classification_report(y_test,y_pred))
	print(accuracy)
	return accuracy

