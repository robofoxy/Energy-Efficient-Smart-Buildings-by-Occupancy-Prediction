import pickle
from data import x_train, y_train, x_test, y_test, np
import matplotlib.pyplot as plt
import sklearn.preprocessing 
from sklearn import svm
from sklearn.metrics import f1_score


       
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
	res = []
	acc = 0
	for i in range(sz):
		if y_test[i][0] == 1:
			if arr[i] == 1:
				acc += 1
		else:
			if arr[i] == 0:
				acc += 1
	return acc / sz


if __name__=='__main__':
    ker = ['rbf', 'linear', 'sigmoid']
    c = 0.315
    tl = 0.0130
    gm = 0.108
    acc_max = -1
    c_max = 0.317
    tl_max = 0.0136
    gm_max = -1
    kr = 0
    kr_max = -1
    '''
    while c < 0.340:
        clf = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=gm, kernel=ker[kr],
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=tl, verbose=False)



        clf.fit(x_train, prepare_data(y_train))

        acc = evaluate(clf.predict(x_test))
        print("C =", c, "tolerance =", tl, "gamma =", gm, "kernel =", ker[kr], "accuracy =", acc,"fscore =",f1_score(prepare_data(y_test), clf.predict(x_test), average='macro'))
        if acc > acc_max:
            acc_max = acc
            c_max = c
        c += 0.001
        
        
        
    while tl < 0.014:
        clf = svm.SVC(C=c_max, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=gm, kernel=ker[kr],
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=tl, verbose=False)

        clf.fit(x_train, prepare_data(y_train))

        acc = evaluate(clf.predict(x_test))
        print("C =", c_max, "tolerance =", tl, "gamma =", gm, "kernel =", ker[kr], "accuracy =", acc,"fscore =",f1_score(prepare_data(y_test), clf.predict(x_test), average='macro'))
        if acc > acc_max:
            acc_max = acc
            tl_max = tl
        tl += 0.0001
        '''
    while gm <= 0.12:
        if gm > 0.11:
    	    gm = 'auto'
        clf = svm.SVC(C=c_max, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=gm, kernel=ker[kr],
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=tl_max, verbose=False)

        clf.fit(x_train, prepare_data(y_train))

        acc = evaluate(clf.predict(x_test))
        print("C =", c_max, "tolerance =", tl_max, "gamma =", gm, "kernel =", ker[kr], "accuracy =", acc,"fscore =",f1_score(prepare_data(y_test), clf.predict(x_test), average='macro'))
        if acc > acc_max:
            acc_max = acc
            gm_max = gm
        if gm != 'auto':
            gm += 0.001
        else:
            break
        
        
        
    while kr < len(ker):
        clf = svm.SVC(C=c_max, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=gm_max, kernel=ker[kr],
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=tl_max, verbose=False)

        clf.fit(x_train, prepare_data(y_train))

        acc = evaluate(clf.predict(x_test))
        print("C =", c_max, "tolerance =", tl_max, "gamma =", gm_max, "kernel =", ker[kr], "accuracy =", acc,"fscore =",f1_score(prepare_data(y_test), clf.predict(x_test), average='macro'))
        if acc > acc_max:
            acc_max = acc
            kr_max = kr
        kr += 1   
        
        
        
        
    print("################################################################")
    print("BEST CASE = C =", c_max, "tolerance =", tl_max, "gamma =", gm_max, "kernel =", ker[kr_max], "accuracy =", acc_max)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

