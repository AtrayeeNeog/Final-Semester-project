import numpy as np
import pickle
import tensorflow as tf
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def lreluder(x, leak=0.2):
    if (x>0):
        return 1
    else:
        return leak

cs=np.genfromtxt("/home/raj/Desktop/data for project/all_data_edited.csv",delimiter=",",skip_header=1)
#input data
in1=cs[:,3]
in2=cs[:,5]
in3=cs[:,6]
in4=cs[:,7]
in5=cs[:,15]
in6=cs[:,16]
in7=cs[:,17]
in8=cs[:,18]
in9=cs[:,19]
in10=cs[:,20]
in11=cs[:,21]
t=np.matrix([in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]).T
X=np.array(t)

#input data

#X=np.array([t[:,0]],[t[:,1]],[t[:,2]],[t[:,3]],[t[:,4]],[t[:,5]],[t[:,6]],[t[:,7]],[t[:,8]],[t[:,8]],[t[:,9]],[t[:,10]])
'''X=np.array([[0,0,1],
	   [0,1,1],	
	   [1,0,1],
	   [1,1,1],
	   [1,2,1]])'''

#output data
y1=np.array([cs[:,8]])
y=y1.T
np.random.seed(1)


#synapses
syn0 = 2*np.random.random((11,12))-1
syn1 = 2*np.random.random((12,1))-1

 
#training step
def train_model(X,y,syn0,syn1):
	for j in xrange(100000):
	
		l0=X #layer 0:input layer
		l1=lrelu(np.dot(l0, syn0)) #layer 1:hidden layer
		l2=lrelu(np.dot(l1, syn1)) #layer 2:output layer
	
		l2_error = y-l2 
		
		if(j % 1000)==0:   
			print "Error:" + str(np.mean(np.abs(l2_error)))

		l2_delta = l2_error*lreluder(l2)
		
		l1_error = l2_delta.dot(syn1.T)
		
		l1_delta = l1_error*lreluder(l1) 
		
	#update weights
		syn1 +=l1.T.dot(l2_delta)
		syn0 +=l0.T.dot(l1_delta)
	return l2

i=train_model(X,y,syn0,syn1)

with open('ann.pickle','wb') as temp:
	pickle.dump(train_model, temp)
pickle_in = open('ann.pickle','rb')
train_model=pickle.load(pickle_in)

print "Output after training"
print i 
