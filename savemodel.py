#savemodel.py
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import sys
import pickle
rng = numpy.random

class Wrapper(object):
 def __init__(self,sess,fakeid='fake'):
 self.fakeid=fakeid
 self.sess = sess
 self.W = tf.Variable(rng.randn(), name="weight")
 self.b = tf.Variable(0.0, name="bias")

 def change(self,n=0):
 sess.run(self.b.assign(100+n))

 def construct_pred(self,X,Y):
 self.pred_op = tf.add(tf.mul(X, self.W), self.b)
 return(self.pred_op)

 def prediction(self):
 print(self.sess.run(self.W))


# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19])
n_samples = train_X.shape[0]




# Launch the graph

with tf.Graph().as_default() as g:
 X = tf.placeholder("float")
 Y = tf.placeholder("float")
 with tf.Session() as sess:

 model = Wrapper(sess)
 pred = model.construct_pred(X,Y)
 # Mean squared error
 cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
 # Gradient descent
 optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


 saver = tf.train.Saver()
 # Initializing the variables
 init = tf.initialize_all_variables()
 sess.run(init)
 print([v.op.name for v in tf.all_variables()])
 model.change()
 saver.save(sess,'modelmodel.change(n=1)
 print(sess.run(model.b)) #101
 saver.restore(sess,'model')
 print(sess.run(model.b)) #100