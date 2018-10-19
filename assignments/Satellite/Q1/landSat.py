#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from functools import reduce
import random

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
beta = 1e-6
epochs = 1000
batch_size = 32
num_neurons = 10
seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('../../data/sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix


# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

n = trainX.shape[0]


# Create the model
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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
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
    print("Batch size %d"%batch_size)
    num_batches = int(n / batch_size)
    for e in range(epochs):
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
        train_acc.append(reduce(lambda x, y: x+y, accumulated_acc)/len(accumulated_acc))
        train_loss.append(reduce(lambda x, y: x+y, accumulated_loss)/len(accumulated_loss))

        if e % 100 == 0:
            print('iter %d: accuracy %g'%(e, train_acc[-1]))
            print('iter %d: loss %g'%(e, train_loss[-1]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

