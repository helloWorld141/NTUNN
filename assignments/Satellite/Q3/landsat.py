#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import random
from functools import reduce
from timeit import default_timer as timer
import json

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def prepare_data(filepath, X_min=None, X_max=None, delimiter=' '):
    data = np.loadtxt(filepath, delimiter=delimiter)
    X, Y_ = data[:,:-1], data[:,-1].astype(int)
    if X_min is None:
        X_min = np.min(X, axis=0)
    if X_max is None:
        X_max = np.max(X, axis=0)
    X = scale(X, X_min, X_max)
    Y_[Y_ == 7] = 6 
    # convert labels to one hot matrix
    Y = np.zeros((Y_.shape[0], NUM_CLASSES))
    Y[np.arange(Y_.shape[0]), Y_-1] = 1 #one hot matrix
    return (X, Y, X_min, X_max)

#### load config from file
conf = json.loads(open("conf").read())
file = open('log', 'r')
log = file.read()
if len(log) != 0:
    log = json.loads(log)
else:
    log = {"logs": []}
file.close()
log_entry = {
    "batch_size": 0,
    "avg_running_time_1e": 0,
    "min_running_time_1e": 0,
    "max_running_time_1e": 0,
    "running_time": []
}

NUM_FEATURES = 36
NUM_CLASSES = 6
learning_rate = 0.01
beta = 1e-6
epochs = conf["epochs"]
batch_size = conf["batch_size"]
num_neurons = conf['nhidden']
seed = 10
np.random.seed(seed)


trainX, trainY, X_min, X_max = prepare_data('../../data/sat_train.txt')
testX, testY, _, _ = prepare_data('../../data/sat_test.txt', X_min, X_max)

n = trainX.shape[0]
print("Number of examples is: " + str(n))
def createModel(graph=None):
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES]) ### input
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) ### target output
    # Build the graph for the deep net
    w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES)), seed=25), name='weights1')
    b1  = tf.Variable(tf.zeros([num_neurons]), name='biases1')
    hidden = tf.nn.softmax(tf.matmul(x, w1) + b1)
    w2 = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons)), seed=25), name='weights2')
    b2  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases2')
    y  = tf.matmul(hidden, w2) + b2 #### linear ouput of NN, prediction values
    ### defind loss function with L2 regularization
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y) #### softmax activation
    J = tf.reduce_mean(cross_entropy)
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    loss = tf.reduce_mean(J + beta * regularizer)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return (train_op, accuracy, loss, x, y_)

train_op, accuracy, loss, x, y_ = createModel()
sess = tf.Session()
init = tf.global_variables_initializer()
print(trainX.shape)
print(trainY.shape)
whole = np.column_stack((trainX, trainY))
random.shuffle(whole)
trainX = whole[:, :-NUM_CLASSES]
trainY = whole[:, -NUM_CLASSES:]
print(trainX.shape)
print(trainY.shape)
train_acc = []
train_loss = []
test_acc = []
test_loss = []
running_time = []
print("Experiment with {} hidden units".format(num_neurons))
log_entry["batch_size"] = batch_size
log_entry['nhidden'] = num_neurons
name = "{} hidden units".format(num_neurons)
num_batches = int(n / batch_size)
sess.run(init)
for e in range(epochs):
    start_time = timer()
    accumulated_acc = []
    accumulated_loss = []
    for i in range(num_batches):
        start = i*batch_size
        end = (i+1)*batch_size if i < num_batches-1 else n
        batchX = trainX[start: end]
        batchY = trainY[start: end]
        _, acc_, loss_ = sess.run([train_op, accuracy, loss], feed_dict={x: batchX, y_: batchY})
        accumulated_acc.append(acc_)
        accumulated_loss.append(loss_)
    end_time = timer()
    epoch_time = end_time - start_time
    # print(epoch_time)
    running_time.append(epoch_time)
    train_acc.append(reduce(lambda x, y: x+y, accumulated_acc)/len(accumulated_acc))
    train_loss.append(reduce(lambda x, y: x+y, accumulated_loss)/len(accumulated_loss))
    acc_, loss_ = sess.run([accuracy, loss], feed_dict={x: testX, y_: testY})
    test_acc.append(acc_)
    test_loss.append(loss_)
    if e % 100 == 0:
        print('iter %d: Training accuracy %g'%(e, train_acc[-1]))
        print('iter %d: Testing accuracy %g'%(e, test_acc[-1]))
# print(running_time)
log_entry["avg_running_time_1e"] = reduce(lambda x,y: x+y, running_time)/len(running_time)
log_entry["min_running_time_1e"] = min(running_time)
log_entry["max_running_time_1e"] = max(running_time)
log_entry["running_time"] = running_time
# print(running_time)
# plot accuracy curves
plt.figure("Accuracy " + name)
plt.plot(range(epochs), train_acc, label="Training accuracy")
plt.plot(range(epochs), test_acc, label="Testing accuracy")
plt.xlabel(str(epochs) + ' iterations')
plt.legend()
plt.savefig("Figures/Accuracy_{}.png".format(name), bbox_inches='tight')
#plot loss curves
plt.figure("Loss " + name)
plt.plot(range(epochs), train_loss, label="Training loss")
plt.plot(range(epochs), test_loss, label="Testing loss")
plt.xlabel(str(epochs) + ' iterations')
plt.legend()
plt.savefig("Figures/Loss_{}.png".format(name), bbox_inches='tight')
sess.close()
### show 'em bitches ###
#plt.show()
log["logs"].append(log_entry)
with open("log", "w") as file:
    json.dump(log, file)
