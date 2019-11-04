import numpy as np
import matplotlib.pyplot as plt

# N is batch size(sample size); D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4, 11, 12, 1
leak=0.01
'''
# Create random input and output data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
'''
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
x=np.array(t)

#output
y1=np.array([cs[:,8]])
y=y1.T
np.random.seed(1)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 0.00002
loss_col = []
for t in range(200):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, leak*h)  # using ReLU as activate function
    y_pred = h_relu.dot(w2)
    # Compute and print loss
    loss = np.square(y_pred - y).sum() # loss function
    loss_col.append(loss)
    print(t, loss, y_pred)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y) # the last layer's error
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T) # the second laye's error 
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = leak  # the derivate of ReLU
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.plot(loss_col)
plt.show()