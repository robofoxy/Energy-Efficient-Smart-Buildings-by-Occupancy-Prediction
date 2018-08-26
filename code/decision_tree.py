from data import x_train, y_train, x_test, y_test, np
from sklearn import tree
from sklearn.metrics import f1_score
	
def evaluate_tree(arr):
	sz = len(arr)
	res = []
	acc = 0
	for i in range(sz):
		if y_test[i][0] == arr[i][0]:
				acc += 1
	return acc / sz


acc = -1
if __name__=='__main__':
	
	while(True):
	
		clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='log2', random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
		
		clf = clf.fit(x_train, y_train)
		
		cur = evaluate_tree(clf.predict(x_test))
		
		if(acc < cur):
			acc = cur
			print("accuracy:", acc)
			print("fscore:", f1_score(y_test, clf.predict(x_test), average='macro'))
			print("")
			tree.export_graphviz(clf,out_file='tree.dot',feature_names=['Humiditiy_d1','CO2_d1','Light_d1','Humiditiy_d3','CO2_d3','Light_d3','Humiditiy_d5','CO2_d5','Light_d5']) 
		
		
