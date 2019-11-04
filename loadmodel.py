#loadmodel.py
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import sys
rng = numpy.random

#*****************************#
from saver1 import Wrapper
#*****************************#


with tf.Graph().as_default() as g:
 with tf.Session() as sess:
 model = Wrapper(sess)
 with tf.Session() as sess:
 saver = tf.train.Saver()
 saver.restore(sess,'model')
 # Initializing the variables
 print([v.op.name for v in tf.all_variables()])
 print(sess.run(model.b)) #100