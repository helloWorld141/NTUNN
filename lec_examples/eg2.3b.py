#
# Chapter 2, Example 3, using tf.gradients
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

no_iters = 1500
lr = 0.01
SEED = 10
np.random.seed(SEED)

# training data
# generate training data
X = np.random.rand(7,2)
Y = 1.0 +3.3*X[:,0]**2-2.5*X[:,1]+0.2*X[:,0]*X[:,1]
Y = Y.reshape(7,1)

# Model parameters
w = tf.Variable(np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [None, 2])
d = tf.placeholder(tf.float32, [None, 1])

u = tf.matmul(x,w) + b
y = 4*tf.sigmoid(u)-1
loss = tf.reduce_sum(tf.square(d - y)) # sum of the squares

grad_w, grad_b = tf.gradients(loss, [w, b])
w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err = []
for i in range(no_iters):
  sess.run([loss, w_new, b_new], {x: X, d: Y})
  err.append(sess.run(loss, {x: X, d: Y}))

  if (i < 2 or i == no_iters-1):
  	y_, loss_, w_, b_ = sess.run([y, loss, w, b], {x: X, d: Y})
  	print('iter: {}'.format(i+1))
  	print('y: {}'.format(y_))
  	print('loss: {}'.format(loss_))
  	print('w: {}, b: {}'.format(w_, b_))

  if i%100 == 0:
    print('iter: {}, error: {}'.format(i, err[i]))

# evaluate training accuracy
curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x: X, d: Y})
print("w: %s b: %s loss: %s"%(curr_w, curr_b, curr_loss))

plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.savefig('./figures/2.3b_1.png')

plt.figure(2)
plt.plot(X[:,0], X[:,1], 'r.')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title('Input Data')
plt.axis('equal')
plt.savefig('./figures/2.3b_2.png')


pred = sess.run(y, {x: X})

# plot trained and predicted points
fig = plt.figure(3)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y, color='blue', marker='x', label='targets')
ax.scatter(X[:,0], X[:,1], pred, color='red', marker='.', label='predictions')
ax.set_title('Targets and Predictions')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.legend()
plt.savefig('./figures/2.3b_3.png')

X1 = np.arange(-1, 1, 0.1)
X2 = np.arange(-1, 1, 0.1)
X = []
for i in range(len(X1)):
  for j in range(len(X2)):
    X.append([X1[i], X2[j]])

X = np.array(X)
pred = sess.run(y, {x: X})
pred = np.reshape(pred, len(pred))

fig = plt.figure(4)
ax = fig.gca(projection = '3d')
# predicted_surface = ax.plot_surface(X[:,0], X[:,1], pred)
ax.scatter(X[:,0], X[:,1], pred, color='red', marker='.', label='predictions')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
#ax.legend()
plt.savefig('./figures/2.3b_4.png')

plt.show()


