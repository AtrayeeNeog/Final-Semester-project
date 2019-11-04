#!/usr/bin/python

import numpy as np
import tensorflow as tf
import pandas as pd
import easygui as g

j=g.fileopenbox()
df= pd.DataFrame
df= pd.read_csv(j)
#print df
tf.set_random_seed(0)

X1 = df.loc[:,['hrs','Temp','Wind', 'Humidity','Barometer']].as_matrix()
y1= df.loc[:,['flow']].as_matrix()
#df = np.matrix(df)
print X1
#df.to_csv(float_format="%.0f")

a0 = tf.placeholder(tf.float32, [None,5])
y= tf.placeholder(tf.float32, [None, 1])


mid = 6

w10 = tf.Variable(tf.truncated_normal([5,mid]))
b10 = tf.Variable(tf.truncated_normal([1,mid])) 
w21 = tf.Variable(tf.truncated_normal([mid,1]))
b21 = tf.Variable(tf.truncated_normal([1,1]))

def sigma(x):
	return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))
def sigmaprime(x):
	return tf.mul(sigma(x), tf.sub(tf.constant(1.0), sigma(x)))
def relu(x):
	return tf.maximum(0.01*x,x)
def reluprime(x):
        t = tf.zeros(tf.shape(x))
        t1 = tf.ones(tf.shape(x))
        t0 = tf.mul(0.01,t1)

        mask1 = tf.greater(x,t)
        x = tf.select(mask1,t1,x)

        mask2 = tf.less(x,t)
        x = tf.select(mask2,t0,x)

        return x

def Error(x,y):
	return tf.reduce_mean(tf.square(tf.sub(x,y)),0)

#forward prop_for training
z1 = tf.add(tf.matmul(a0,w10),b10)
a1 = sigma(z1)
z2 = tf.add(tf.matmul(a1,w21),b21)
a2 = relu(z2)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print sess.run(a2, feed_dict = {a0: X1, y : y1})

#back prop
diff = tf.sub(a2,y)
p = tf.Variable(0)
p = diff

dz2  = tf.mul(diff,reluprime(z2))
db21  = dz2
dw21 = tf.matmul(tf.transpose(a1),dz2)

da1  = tf.matmul(dz2 , tf.transpose(w21))
dz1 = tf.mul(da1, sigmaprime(z1))
db10 = dz1
dw10 = tf.matmul(tf.transpose(a0),dz1)

#update
eta = tf.constant(0.5)
epoch = 10000

step = [tf.assign(w10,tf.sub(w10, tf.mul(eta, dw10))), tf.assign(b10,tf.sub(b10, tf.mul(eta,tf.reduce_mean(db10, 0)))), tf.assign(w21,tf.sub(w21, tf.mul(eta, dw21))), tf.assign(b21,tf.sub(b21, tf.mul(eta,tf.reduce_mean(db21, 0))))]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(epoch):
		sess.run(step, feed_dict = {a0 : X1, y : y1})
		if i == 1:
			print '@@@@@@@@@@@@'
			print sess.run(a2, feed_dict={a0 : X1})
			print '\n@@@@@@@@@@@'
		if i % 1000 == 0:
			print sess.run(diff, feed_dict = {a0 : X1, y : y1})
		if i == 9990:
                        print '@@@@@@@@@@@@'
                        xyz = sess.run(a2, feed_dict={a0 : X1})
			print 'ERROR ********->',sess.run(Error(xyz,y), feed_dict= {y: y1})
                        print '\n@@@@@@@@@@@'


	msg = "TRAINING COMPLETE!.Do you want to TEST?"
	title = "Please Confirm"
	if g.ccbox(msg, title):# show a Continue/Cancel dialog
		pass # user chose Continue
	else: # user chose Cancel
		sys.exit(0)

	msg = "Enter the data"
	title = "Tensor Flow weather"
	fieldNames = ['hrs','temp','wind','humidity','barometer','visibilty']
	fieldValues = []  # we start with blanks for the values
	fieldValues = g.multenterbox(msg,title, fieldNames)

	# make sure that none of the fields was left blank
	while 1:
		if fieldValues == None: break
		errmsg = ""
		for i in range(len(fieldNames)):
			if fieldValues[i].strip() == "":
				errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
		if errmsg == "": break # no problems found
		fieldValues = g.multenterbox(errmsg, title, fieldNames, fieldValues)
	fV = [float(i) for i in fieldValues]
	print "Reply was:", fV		
	fV = np.matrix(fV)
	ou=sess.run(a2, feed_dict = { a0 : fV})
	print 'OUTPUT =>', ou

ou = ou.tolist()
max1=max(ou)
'''
max2=max(max1)
index1=max1.index(max2)

label = {0:'foggy', 1:'Partly sunny', 2:'scattered clouds', 3: 'passing clouds', 4:'Haze', 5:'overcast', 6:'sunny', 7:'drizzle', 8:'clear', 9:'thunder showers, passing clouds', 10:'thunder showrs, overcast', 11: 'rain', 12: 'hail'}
label = label[index1]
'''
g.msgbox('Predicted weather is %s'%label)

