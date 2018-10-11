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

conf = json.loads(open("conf").read())
NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
beta = 1e-6
epochs = 100
batch_size = conf["batch_size"]
num_neurons = 10
seed = 10
np.random.seed(seed)


trainX, trainY, X_min, X_max = prepare_data('../data/sat_train.txt')
testX, testY, _, _ = prepare_data('../data/sat_test.txt', X_min, X_max)


# experiment with small datasets
#trainX = trainX[:1000]
#trainY = trainY[:1000]

n = trainX.shape[0]
print("Number of examples is: " + str(n))
def createModel(graph=None):
    # with graph.as_default():
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    # Build the graph for the deep net
    weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES)), seed=25), name='weights')
    biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits  = tf.matmul(x, weights) + biases
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    J = tf.reduce_mean(cross_entropy)
    regularizer = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(J + beta * regularizer)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return (weights, train_op, accuracy, loss, x, y_)
log = open("log", "a")
# g = tf.Graph()
weights, train_op, accuracy, loss, x, y_ = createModel()
sess = tf.Session()
init = tf.global_variables_initializer()
for batch_size in [batch_size]:
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
    print("Experiment with batch size %d"%batch_size)
    log.write("Experiment with batch size %d \n"%batch_size)
    name = "batch_of_" + str(batch_size)
    num_batches = int(n / batch_size)
    # g = tf.Graph()
    # train_op, accuracy, loss, x, y_ = createModel(g)
    # re-init the variables
    sess.run(init)
    sess.run(weights)
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
    log.write("Average running time for 1 epoch: {:.5f}s\n".format(reduce(lambda x,y: x+y, running_time)/len(running_time)))
    log.write("Minimum running time for 1 epoch: {:.5f}s\n".format(min(running_time)))
    log.write("Maximum running time for 1 epoch: {:.5f}s\n".format(max(running_time)))
    log.write(str(running_time))
    log.write("\n\n")
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
plt.savefig("Figures/Running_time.png", bbox_inches='tight')
### show 'em bitches ###
plt.show()
log.close()
