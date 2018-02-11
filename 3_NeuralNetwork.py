import numpy as np
import matplotlib.pylab as plt

def StepFunction(x):
    return np.array(x > 0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = StepFunction(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print("sigmoid(x) ", sigmoid(x))

t = np.array([1.0, 2.0, 3.0])
print("1.0 + t ", 1.0 + t)
print("1.0 / t ", 1.0 / t)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
print("np.ndim(A) ", np.ndim(A))
print("A.shape ", A.shape)
print("A.shape[0] ", A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print("B ", B)
print("np.dim(B) ", np.ndim(B))
print("B.shape ", B.shape)

A = np.array([[1, 2], [3, 4]])
print("A.shape ", A.shape)
B = np.array([[5, 6], [7, 8]])
print("B.shape ", B.shape)
print("np.dot(A, B) ", np.dot(A, B))

A = np.array([[1, 2, 3], [4, 5, 6]])
A.shape
B = np.array([[1, 2], [3, 4], [5, 6]])
print("B.shape ", B.shape)
print("np.dot(A, B) ", np.dot(A, B))

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

def softmax(a):
    c = np.exp(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a

    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print("y ", y)
print("np.sum(y) ", np.sum(y))

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

