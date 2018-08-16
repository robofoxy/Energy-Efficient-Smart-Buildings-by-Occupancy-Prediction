import pickle
import tensorflow as tf
import pandas as pd
from data import x_train, y_train, x_test, y_test,np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.metrics
from mpl_toolkits.mplot3d import Axes3D

max_accuracy = 0.0
max_fscore = 0.0
max_lr = 0.0
max_epoch = 0
# A function to create a fully connected layer
def add_layer(input, in_size, out_size, activation_function):
    # Choosing non-random weights of 0.1
    weights = tf.Variable(tf.random_normal([in_size, out_size],seed=2354))
    biases = tf.Variable(tf.zeros([1, out_size]))
    # Calculating Weights and Biases
    wx_plus_b = tf.matmul(input, weights) + biases
    # Using Activation Function
    output = activation_function(wx_plus_b)
    return output

# Place holders for data

def plot(loss_history,accuracies,f1_scores):
    plt.axis([0,8000,0,1.0])
    plt.plot(loss_history,label='loss')
    iter_ = np.arange(250,8250,250)
    plt.plot(iter_,accuracies,label='accuracy')
    plt.plot(iter_,f1_scores,label='f1_scores')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.legend(bbox_to_anchor=(1.125, 1),prop={'size': 6})
    plt.savefig('mix_plot.png')

def print_ranges(arr): 
    for i in range(0,arr.shape[1]):
        print("Attribute %d -> %.3f - %.3f"%(i,np.max(arr[:,i]),np.min(arr[:,i])))
   

def train_test(lr):
    global max_accuracy,max_epoch,max_fscore,max_lr
    xs = tf.placeholder(tf.float32, shape=[None, 9])
    ys = tf.placeholder(tf.float32, shape=[None, 2])
    dop = tf.placeholder(tf.float32) # Dropout Prob

    
    l1 = add_layer(xs, in_size=9, out_size=10, activation_function=tf.nn.sigmoid)

    l2 = add_layer(l1, in_size=10, out_size=10, activation_function=tf.nn.sigmoid)
    # Drop out
    l2 = tf.nn.dropout(l2, dop)

    l3 = add_layer(l2, in_size=10, out_size=8, activation_function=tf.nn.sigmoid)

    l4 = add_layer(l3,in_size=8,out_size=4,activation_function=tf.nn.sigmoid )

    # Output layer / Activation function: softmax
    output = add_layer(l4, in_size=4, out_size=2, activation_function=tf.nn.softmax)

    # Loss / Cross entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output), reduction_indices=[1]))

    # Optimizer / Gradient Descent
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    # A list to keep all losses in
    all_losses = []
    accuracies = []
    f1_scores = []
    # Creating TensorFlow Session
    with tf.Session() as sess:
        sess = tf.Session()

        # Initializing Variables
        sess.run(tf.global_variables_initializer())

        # Training
        for epoch in range(8000):
            sess.run(train_step, feed_dict={xs: x_train, ys: y_train, dop: 1})
            epoch_loss = sess.run(cross_entropy, feed_dict={xs: x_test, ys: y_test, dop: 1})
            all_losses.append(epoch_loss)
            if (epoch+1)%100==0:
                print("Epoch Number: %d Loss Value: %f" % (epoch + 1, epoch_loss))

            
            if (epoch+1)%250==0:
                hypothesis = sess.run(output, feed_dict={xs: x_test, dop: 1})
                pred = tf.equal(tf.argmax(y_test,1),tf.argmax(hypothesis,1))
                accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))
                test_out = accuracy.eval({xs:x_test,ys:y_test})
                f1_score = sklearn.metrics.f1_score(np.argmax(y_test,1),np.argmax(hypothesis,1))
                if test_out > max_accuracy:
                    max_accuracy = test_out
                    max_fscore = f1_score
                    max_lr = lr
                    max_epoch = epoch+1
                
                accuracies.append(test_out)
                f1_scores.append(f1_score)
        sess.close()

    print(accuracies)
    print(f1_scores)
    return all_losses,accuracies,f1_scores

def normalize():
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    return x_train_minmax


def naive_2viz(occupied,non_occupied):
    # Apply normalization onto [0-1] range
    
    x_train = normalize()
    for i in range(0,x_train.shape[1]):
        for j in range(i+1,x_train.shape[1]):
            plt.scatter(x_train[occupied][:,i],x_train[occupied][:,j],c='green')
            plt.scatter(x_train[non_occupied][:,i],x_train[non_occupied][:,j],c='red')
            fname=str(i)+":"+str(j)+"viz.jpg"
            plt.savefig(fname)


def naive_3viz(occupied,non_occupied):
    # Apply normalization onto [0-1] range
    
    x_train = normalize()
    for i in range(0,x_train.shape[1]):
        for j in range(i+1,x_train.shape[1]):
            for k in range(j+1,x_train.shape[1]):
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(x_train[non_occupied][:,i],x_train[non_occupied][:,j],x_train[non_occupied][:,k],c='red')
                ax.scatter(x_train[occupied][:,i],x_train[occupied][:,j],x_train[occupied][:,k],c='green')
                fname=str(i)+":"+str(j)+":"+str(k)+"viz.jpg"
                plt.savefig(fname)
def best_2features_viz(occupied,non_occupied):
    # Apply normalization onto [0-1] range
    
    x_train = normalize()
    x_t = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2,k=2).fit_transform(x_train,y_train)
    plt.scatter(x_t[occupied][:,0],x_t[occupied][:,1],c='green')
    plt.scatter(x_t[non_occupied][:,0],x_t[non_occupied][:,1],c='red')
    plt.savefig('best_2features.jpg')

def best_3features_viz(occupied,non_occupied):
    # Apply normalization onto [0-1] range
    
    x_train = normalize()
    x_t = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2,k=3).fit_transform(x_train,y_train)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_t[non_occupied][:,0],x_t[non_occupied][:,1],x_t[non_occupied][:,2],c='red')
    ax.scatter(x_t[occupied][:,0],x_t[occupied][:,1],x_t[occupied][:,2],c='green')
    angles = np.arange(0,380,20)
    for angle in angles:
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    plt.savefig('best_3features.jpg')

def experiment_lr():
    global max_accuracy,max_epoch,max_fscore,max_lr
    learning_rates = np.arange(0.001,0.011,0.001)
    plt.axis([0,8000,0,1])
    i = 1
    iter_ = np.arange(500,8500,500)
    for lr in learning_rates: 
        losses,accuracies,f1_scores = train_test(lr)
        
        plt.plot(losses,label='loss'+str(i))
        
        plt.plot(iter_,accuracies,label='accuracy'+str(i))
        plt.plot(iter_,f1_scores,label='fscores'+str(i))
        i+=1
    
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.legend(bbox_to_anchor=(1.125, 1),prop={'size': 6})
    plt.savefig('mix_plot.png')
    
if __name__=='__main__':
    occupied = np.where(y_train[:,0]==1)[0]
    non_occupied = np.where(y_train[:,0]==0)[0]
    print("Occupied #: %d"%(occupied.shape[0]))
    print("Non-occupied #: %d"%(non_occupied.shape[0]))
    #--------------------DATA VISUALIZATION--------------------------------#
    
    #print_ranges(x_train)
    # Naive 2 feature Method
    #naive_2viz(occupied,non_occupied)
    
    # Naive 3 feature Method
    #naive_3viz(occupied,non_occupied)
    # Choose best 2 univariate features from the dataset so as to visualize the data
    #best_2features_viz(occupied,non_occupied)
    
    # Choose best 3 univariate features from the dataset so as to visualize the data
    #best_3features_viz(occupied,non_occupied)
    #---------------------------------------------------------------------#

    # Experiment with a range of learning rates
    #experiment_lr()

    lr = 0.003
    l,a,f = train_test(lr)
    plot(l,a,f)
    print("Maximum Accuracy: %.4f, F1 Score: %.4f with learning rate: %.3f at epoch %d"%(max_accuracy,max_fscore,max_lr,max_epoch))
