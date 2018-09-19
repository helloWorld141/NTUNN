#
# Chapter 1, Example 1
#

import tensorflow as tf


# build the computational graph
w = tf.Variable([2.5, -0.2, 1.0], tf.float32) # trainable parameters
b = tf.Variable(-0.5, tf.float32)

x = tf.placeholder(tf.float32)
u = tf.tensordot(w, x, axes=1) + b
y = 0.8/(1+tf.exp(-1.2*u))

# evaluate the computational graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run([u, y], {x: [0.8,2.0, -0.5]}))
print(sess.run([u, y], {x: [-0.4,1.5, 1.0]}))
