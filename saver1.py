import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import sys
import pickle
import easygui as gui

j=gui.fileopenbox()
df= pd.DataFrame
df= pd.read_csv(j)
rng = numpy.random
tf.set_random_seed(0)


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

class Wrapper(object):
 def __init__(self,sess,fakeid='fake'):
  self.fakeid=fakeid
  self.sess = sess
  self.w10 = tf.Variable(tf.truncated_normal([6,mid]))
  self.b10 = tf.Variable(tf.truncated_normal([1,mid])) 
  self.w21 = tf.Variable(tf.truncated_normal([mid,mid]))
  self.b21 = tf.Variable(tf.truncated_normal([1,mid]))
  self.w31 = tf.Variable(tf.truncated_normal([mid,1]))
  self.b31 = tf.Variable(tf.truncated_normal([1,1]))


  def change(self,n=0):
    sess.run(self.b10.assign(100+n))
    sess.run(self.b21.assign(100+n))
    sess.run(self.b31.assign(100+n))

  def construct_pred(self,X,Y):
    self.pred_op = step = [tf.assign(w10,tf.sub(w10, tf.mul(eta, dw10))), tf.assign(b10,tf.sub(b10, tf.mul(eta,tf.reduce_mean(db10, 0)))), tf.assign(w21,tf.sub(w21, tf.mul(eta, dw21))), tf.assign(b21,tf.sub(b21, tf.mul(eta,tf.reduce_mean(db21, 0)))),tf.assign(w31,tf.sub(w31, tf.mul(eta, dw31))), tf.assign(b31,tf.sub(b31, tf.mul(eta,tf.reduce_mean(db31, 0))))]

    return(self.pred_op)

 def prediction(self):
  print(self.sess.run(self.W))


# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
X1 = df.loc[:,['hrs','Temp','Wind', 'Humidity','Barometer','Visibility']].as_matrix()
normalize = max(df['flow'])
y1= (df.loc[:,['flow']].as_matrix())/normalize
#df = np.matrix(df)
print X1




# Launch the graph

with tf.Graph().as_default() as g:
 a0 = tf.placeholder(tf.float32, [None,6])
 y= tf.placeholder(tf.float32, [None, 1])
 iterate = tf.placeholder(tf.float32)
 with tf.Session() as sess:

    model = Wrapper(sess)
    pred = model.construct_pred(X,Y)
    # Mean squared error
    cost = tf.reduce_mean(tf.square(tf.sub(x,y)),0)
    # Gradient descent
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
    ou=sess.run(a3, feed_dict = { a0 : fV})
    ou = max(normalize*ou)
    print 'OUTPUT =>', abs(ou)
 saver = tf.train.Saver()
 # Initializing the variables
 init = tf.initialize_all_variables()
 sess.run(init)
 print([v.op.name for v in tf.all_variables()])
 model.change()
 saver.save(sess,'modelmodel.change(n=1)
 print(sess.run(model.b10))#101
 print(sess.run(model.b21))#101
 print(sess.run(model.b31))#101
 saver.restore(sess,'model')
 print(sess.run(model.b10)) #100
 print(sess.run(model.b21))#101
 print(sess.run(model.b3))#101