import pickle
from data import x_train, y_train, x_test, y_test, np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def prepare_data(arr):
	sz = len(arr)
	res = []
	for i in range(sz):
		if arr[i][0] == 1:
			res.append(1)
		else:
			res.append(0)
	
	return res

def evaluate(arr):
	sz = len(arr)
	acc = 0
	for i in range(sz):
		if y_test[i][0] == arr[i]:
				acc += 1
	return acc / sz

if __name__=='__main__':
	kernel = 'rbf'
	best_c_closed = 0.225 # Best C found in Grid Search stage
	best_c_open = 0.2375 # Best C found in Grid Search stage
	best_gamma_closed = 1.6 # Best gamma found in Grid Search stage
	best_gamma_open = 2.8 # Best gamma found in Grid Search stage
	tl = 0.0147
	scaler = preprocessing.StandardScaler().fit(x_train)
	# As proposed in SVM Guide, use the same linear scaling for both training and test inputs.
	x_scaled_train = scaler.transform(x_train)
	x_scaled_test = scaler.transform(x_test)
	clf = svm.SVC(C=best_c_closed, cache_size=400, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=2, gamma=best_gamma_closed, kernel=kernel,
        max_iter=-1, probability=False, random_state=None, shrinking=True, verbose=False,tol=tl)
	clf.fit(x_scaled_train,prepare_data(y_train))
	acc = evaluate(clf.predict(x_scaled_test))
	print("Accuracy: %lf,F1: %lf"%(acc,f1_score(prepare_data(y_test), clf.predict(x_scaled_test), average='macro')))
	
	