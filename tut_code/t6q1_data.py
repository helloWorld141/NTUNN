import numpy as np
import tensorflow as tf
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

no_features = 2    

seed = 10
np.random.seed(seed)

# generate training data
X = np.zeros((10*10, no_features))
no_data = 0
for i in np.arange(-1.0, 1.001, 2.0/9.0):
    for j in np.arange(-1.0, 1.001, 2.0/9.0):
        X[no_data] = [i, j]
        no_data += 1

Y = np.zeros((no_data, 1))
Y[:,0] = np.sin(np.pi*X[:,0])*np.cos(2*np.pi*X[:,1])

idx = np.arange(no_data)
np.random.shuffle(idx)
X, Y = X[idx], Y[idx]
Xtrain, Ytrain, Xtest, Ytest = X[:70], Y[:70], X[70:], Y[70:]

# plot trained and predicted points
plt.figure(1)
plt.plot(Xtrain[:,0], Xtrain[:,1], 'b.', label='train')
plt.plot(Xtest[:,0], Xtest[:,1], 'rx', label='test')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('test and train inputs')
plt.legend()
plt.savefig('./figures/t6q1_data_1.png')

plt.show()
