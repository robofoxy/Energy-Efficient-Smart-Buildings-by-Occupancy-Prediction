import pickle
from data import x_train, y_train, x_test, y_test, np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import threading


global good_c
global good_gamma
global max_accuracy

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

def normalize():
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    return x_train_minmax

class CrossValidationThread (threading.Thread):
	def __init__(self,c,gamma,x):
		threading.Thread.__init__(self)
		self.c = c
		self.gamma = gamma
		self.x = x
	def run(self):
		global good_c,good_gamma,max_accuracy
		clf = svm.SVC(C=self.c,class_weight=None,decision_function_shape='ovr',gamma=self.gamma)
		total_accuracy = 0.0
		for train,test in kf.split(self.x):
			clf.fit(self.x[train],prepare_data(y_train[train]))
			acc = evaluate(clf.predict(self.x[test]))
			total_accuracy += acc
		threadLock.acquire(timeout=5)
		if total_accuracy > max_accuracy:
			max_accuracy = total_accuracy
			good_c = self.c
			good_gamma = self.gamma
		threadLock.release()

threadLock = threading.Lock()
			
if __name__=='__main__':
	global good_c,good_gamma,max_accuracy
	kernel = 'rbf'
	scaler = preprocessing.StandardScaler().fit(x_train)
	# As proposed in SVM Guide, use the same linear scaling for both training and test inputs.
	x_scaled_train = scaler.transform(x_train)
	x_scaled_test = scaler.transform(x_test)

	# Need to do cross validation. Let's choose v = 17, since training set consists of 8143 instances. Each fold will have 479 instances.
    # Coarse search
	C = [2**exp for exp in range(-5,15,2)]
	Gamma =[2**exp for exp in range(-15,3,2)]
	kf = KFold(n_splits=17)
	good_c = None
	good_gamma = None
	max_accuracy = 0.0
	for i in range(0,len(C)):
		threads = []
		for j in range(0,len(Gamma)):
			#print("At thread with C= %lf, Gamma = %lf"%(C[i],Gamma[j]))
			thread = CrossValidationThread(C[i],Gamma[j],x_scaled_train)
			thread.start()
			threads.append(thread)
		for t in threads:
			t.join(timeout=150)
			if t.isAlive():
				print("Thread with C= %lf, Gamma = %lf is timed out"%(t.c,t.gamma))
		threads.clear()
	print("Proceeding with Grid Search")
	threadLock = threading.Lock()
    # Grid search
    if good_c - 0.5>0:
    	min_c = good_c - 0.5
    else:
    	min_c = 0
	if good_gamma - 0.5>0:
		min_gamma = good_gamma - 0.5
	else:
		min_gamma = 0
	C = [2**exp for exp in np.arange(min_c,good_c+0.5,0.1)]
	Gamma = [2**exp for exp in np.arange(min_gamma,good_gamma+0.5,0.1)]
	best_c = None
	best_gamma = None
	for i in range(0,len(C)):
		threads = []
		for j in range(0,len(Gamma)):
			#print("At thread with C= %lf, Gamma = %lf"%(C[i],Gamma[j]))
			thread = CrossValidationThread(C[i],Gamma[j],x_scaled_train)
			thread.start()
			threads.append(thread)
		for t in threads:
			t.join()
			if t.isAlive():
				print("Thread with C= %lf, Gamma = %lf is timed out"%(t.c,t.gamma))
		threads.clear()
	best_c = good_c
	best_gamma = good_gamma
	print("Best parameters: %lf,%lf"%(best_c,best_gamma))

       
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
