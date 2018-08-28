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
	c = 0.125000 # Best C found in CV stage
	gamma = 2.000000 # Best gamma found in CV stage
	
	scaler = preprocessing.StandardScaler().fit(x_train)
	# As proposed in SVM Guide, use the same linear scaling for both training and test inputs.
	x_scaled_train = scaler.transform(x_train)
	x_scaled_test = scaler.transform(x_test)
	
	C =  np.arange(c-c,c+c,c/5)
	Gamma = np.arange(gamma-gamma,gamma+gamma,gamma/5)
	for i in range(1,len(C)):
		for j in range(1,len(Gamma)):
			clf = svm.SVC(C=C[i], cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=Gamma[j], kernel=kernel,
        max_iter=-1, probability=True, random_state=None, shrinking=True, verbose=False)
			clf.fit(x_scaled_train,prepare_data(y_train))
			acc = evaluate(clf.predict(x_scaled_test))
			print("C = %lf, Gamma = %lf => Accuracy: %lf,F1: %lf"%(C[i],Gamma[j],acc,f1_score(prepare_data(y_test), clf.predict(x_scaled_test), average='macro')))
			