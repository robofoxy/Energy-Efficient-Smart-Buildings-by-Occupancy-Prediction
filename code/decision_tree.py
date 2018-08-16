import pickle
import tensorflow as tf
import pandas as pd
from data import x_train, y_train, x_test, y_test, np
import matplotlib.pyplot as plt
import sklearn.preprocessing 
from sklearn import svm
from sklearn import tree
from sklearn.metrics import f1_score

# A function to create a fully connected layer
def add_layer(input, in_size, out_size, activation_function):
    # Choosing non-random weights of 0.1
    weights = tf.Variable(tf.random_normal([in_size, out_size],seed=2314))
    biases = tf.Variable(tf.zeros([1, out_size]))
    # Calculating Weights and Biases
    wx_plus_b = tf.matmul(input, weights) + biases
    # Using Activation Function
    output = activation_function(wx_plus_b)
    return output

# Place holders for data

def plot(loss_history):
    plt.plot(loss_history)
    plt.savefig('loss_plot.png')


   

def train():
    xs = tf.placeholder(tf.float32, shape=[None, 9])
    ys = tf.placeholder(tf.float32, shape=[None, 2])
    dop = tf.placeholder(tf.float32) # Dropout Prob

    # Hidden Layer 1 / Activation function: rel
    l1 = add_layer(xs, in_size=9, out_size=10, activation_function=tf.nn.sigmoid)

    l2 = add_layer(l1, in_size=10, out_size=10, activation_function=tf.nn.sigmoid)

    # Drop out
    l2 = tf.nn.dropout(l2, dop)

    l3 = add_layer(l2, in_size=10, out_size=8, activation_function=tf.nn.sigmoid)


    # Output layer / Activation function: softmax
    output = add_layer(l3, in_size=8, out_size=2, activation_function=tf.nn.softmax)

    # Loss / Cross entropy
    #MSE = tf.reduce_mean(tf.square(ys-output))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output), reduction_indices=[1]))

    # Optimizer / Gradient Descent
    train_step = tf.train.AdamOptimizer(learning_rate=.001).minimize(cross_entropy)

    # A list to keep all losses in
    all_losses = []

    # Creating TensorFlow Session
    with tf.Session() as sess:
        sess = tf.Session()

        # Initializing Variables
        sess.run(tf.global_variables_initializer())

        # Training
        for epoch in range(8000):
            sess.run(train_step, feed_dict={xs: x_train, ys: y_train, dop: 1})
            epoch_loss = sess.run(cross_entropy, feed_dict={xs: x_test, ys: y_test, dop: 1})
            if epoch % 10 == 0:
                all_losses.append(epoch_loss)
                print("Epoch Number: %d Loss Value: %f" % (epoch + 1, epoch_loss))

        hypothesis = sess.run(output, feed_dict={xs: x_test, dop: 1})
        pred = tf.equal(tf.argmax(y_test,1),tf.argmax(hypothesis,1))
        accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))
        test_out = accuracy.eval({xs:x_test,ys:y_test})
        print("Test Accuracy: %.3f"%(test_out))
        sess.close()

    plot(all_losses)

def print_ranges():
    np_x_train = np.array(x_train)
    for i in range(0,np_x_train.shape[1]):
        print("Attribute %d -> %.3f - %.3f"%(i,np.max(np_x_train[:,i]),np.min(np_x_train[:,i])))
        
        
        
        
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
	
	
	
def evaluate_tree(arr):
	sz = len(arr)
	res = []
	acc = 0
	for i in range(sz):
		if y_test[i][0] == arr[i][0]:
				acc += 1
	return acc / sz


if __name__=='__main__':
	'''
    
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)



    clf.fit(x_train, prepare_data(y_train))
    
    
    print(evaluate(clf.predict(x_test)))
	


	clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=6, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=2, random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
	'''
	
	clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=6, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='log2', random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
	
	clf = clf.fit(x_train, y_train)
	print(evaluate_tree(clf.predict(x_test)))
	print("fscore:",f1_score(y_test, clf.predict(x_test), average='macro'))
	tree.export_graphviz(clf,out_file='tree.dot',feature_names=['Humiditiy_d1','CO2_d1','Light_d1','Humiditiy_d3','CO2_d3','Light_d3','Humiditiy_d5','CO2_d5','Light_d5']) 
		
		
