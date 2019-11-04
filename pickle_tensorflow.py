#!/usr/bin/python

import numpy as np
import tensorflow as tf
import pandas as pd
import easygui as g
import pickle
j=g.fileopenbox()
df= pd.DataFrame
df= pd.read_csv(j)
n1=np.array([])
n2=np.array([])
n3=np.array([])

#print df
tf.set_random_seed(0)
X1 = df.loc[:688,['hrs','Temp','Wind', 'Humidity','Barometer']].as_matrix()
X2 = df.loc[688:,['hrs','Temp','Wind', 'Humidity','Barometer']].as_matrix()

normalize = max(df['flow'])
y1= (df.loc[:688,['flow']].as_matrix())/normalize
y2= (df.loc[688:,['flow']].as_matrix())/normalize

#df = np.matrix(df)
print X1
#df.to_csv(float_format="%.0f")

a0 = tf.placeholder(tf.float32, [None,5])
y= tf.placeholder(tf.float32, [None, 1])
iterate = tf.placeholder(tf.float32)


mid = 6

w10 = tf.Variable(tf.truncated_normal([5,mid]))
b10 = tf.Variable(tf.truncated_normal([1,mid])) 
w21 = tf.Variable(tf.truncated_normal([mid,mid]))
b21 = tf.Variable(tf.truncated_normal([1,mid]))
w31 = tf.Variable(tf.truncated_normal([mid,1]))
b31 = tf.Variable(tf.truncated_normal([1,1]))


def sigma(x):
	return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))
def sigmaprime(x):
	return tf.mul(sigma(x), tf.sub(tf.constant(1.0), sigma(x)))
def relu(x):
	return tf.maximum(0.001*x,x)
def reluprime(x):
        t = tf.zeros(tf.shape(x))
        t1 = tf.ones(tf.shape(x))
        t0 = tf.mul(0.001,t1)

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
a2 = sigma(z2)
z3 = tf.add(tf.matmul(a2,w31),b31)
a3 = relu(z3)
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print sess.run(a2, feed_dict = {a0: X1, y : y1})

#back prop
diff = tf.sub(a3,y)
p = tf.Variable(0)
p = diff

dz3  = tf.mul(diff,reluprime(z3))
db31  = dz3
dw31 = tf.matmul(tf.transpose(a2),dz3)
da2  = tf.matmul(dz3 , tf.transpose(w31))
dz2  = tf.mul(da2,sigmaprime(z2))
db21  = dz2
dw21 = tf.matmul(tf.transpose(a1),dz2)

da1  = tf.matmul(dz2 , tf.transpose(w21))
dz1 = tf.mul(da1, sigmaprime(z1))
db10 = dz1
dw10 = tf.matmul(tf.transpose(a0),dz1)

#update
eta = 0.15/(1+(iterate/1000))
epoch = 20000

step = [tf.assign(w10,tf.sub(w10, tf.mul(eta, dw10))), tf.assign(b10,tf.sub(b10, tf.mul(eta,tf.reduce_mean(db10, 0)))), tf.assign(w21,tf.sub(w21, tf.mul(eta, dw21))), tf.assign(b21,tf.sub(b21, tf.mul(eta,tf.reduce_mean(db21, 0)))),tf.assign(w31,tf.sub(w31, tf.mul(eta, dw31))), tf.assign(b31,tf.sub(b31, tf.mul(eta,tf.reduce_mean(db31, 0))))]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(epoch):
		sess.run(step, feed_dict = {a0 : X1, y : y1,iterate : i})
		if i == 1:
			print '@@@@@@@@@@@@'
			print sess.run(a3, feed_dict={a0 : X1})
			print '\n@@@@@@@@@@@'
		if i % 1000 == 0:
			print sess.run(diff, feed_dict = {a0 : X1, y : y1})
		if i == 19990:
                        print '@@@@@@@@@@@@'
                        xyz = sess.run(a3, feed_dict={a0 : X1})
			print 'ERROR AFTER TRAINING********->',sess.run(Error(xyz,y), feed_dict= {y: y1})
                        print '\n@@@@@@@@@@@'
	print '\nCROSS VALIDATION ERROR AFTER TESTING-', sess.run(Error(a3,y), feed_dict = {a0 : X2, y : y2})
	n1=sess.run(w10)
	n2=sess.run(w21)
	n3=sess.run(w31)
	with open('file.pkl','w') as f:
  	  pickle.dump(n1, f)
	  pickle.dump(n2, f)
	  pickle.dump(n3, f)

	msg = "TRAINING COMPLETE!.Do you want to TEST?"
	title = "Please Confirm"
	if g.ccbox(msg, title):# show a Continue/Cancel dialog
		pass # user chose Continue
	else: # user chose Cancel
		sys.exit(0)

	msg = "Enter the data"
	title = "Tensor Flow weather"
	fieldNames = ['hrs','temp','wind','humidity','barometer']
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
	ou=sess.run(a3, feed_dict = { a0 : fV})
	ou = max(ou)
	print 'OUTPUT =>', abs(ou)
'''
ou = ou.tolist()
max1=max(ou)
max2=max(max1)
index1=max1.index(max2)

label = {0:'foggy', 1:'Partly sunny', 2:'scattered clouds', 3: 'passing clouds', 4:'Haze', 5:'overcast', 6:'sunny', 7:'drizzle', 8:'clear', 9:'thunder showers, passing clouds', 10:'thunder showrs, overcast', 11: 'rain', 12: 'hail'}
label = label[index1]
'''
g.msgbox('Predicted water requirement is %s'%abs(ou))


