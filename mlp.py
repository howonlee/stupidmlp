#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.

# shamelessly taken and modified by Howon Lee in 2016, cuz BSD allows that
# only Howon doesn't know which version of the BSD license Howon should replicate
# so whatever version NP Rougier wanted, consider it replicated
# dont think that NP Rougier has heard of Howon, because he prolly hasn't
# nor vice versa, now that Howon thinks about it
# -----------------------------------------------------------------------------

# This is an implementation of the multi-layer perceptron with retropropagation
# learning.

# modified by Howon Lee to make a point about the fractal nature of backprop
# play with the params as you like
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import cPickle
import collections
import time
import random
import gzip

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

### don't even ask why we're implementing our own dok
### yah, I know about scipy dok
### it sucks real bad and this is faster for the purposes I want, OK?

def ourdok_ones(size):
    ourdok = collections.defaultdict(float)
    for x in xrange(size):
        ourdok[x] = 1.0
    return ourdok

def ourdok_random(fst_axis, snd_axis):
    ourdok = collections.defaultdict(dict)
    for x in xrange(fst_axis):
        for y in xrange(snd_axis):
            ourdok[x][y] = (2 * random.random() - 1) * 0.00001

# self.layers[i] = ourdok_sigmoid(ourdok_vecmatdot(self.layers[i-1],self.weights[i-1]))

def ourdok_sigmoid(dok):
    return {k: sigmoid(v) for k,v in dok.iteritems()}

def ourdok_vecmatdot(vec, mat):
    new_vec = collections.defaultdict(float)
    for vec_key, vec_val in vec.iteritems():
        for mat_rowkey, mat_row in mat.iteritems():
            for mat_colkey, mat_val in mat_row.iteritems():
                if vec_key == mat_colkey:
                    new_vec[vec_key] += vec[mat_rowkey] * mat_val
    return new_vec

class MLP:
    '''
    Multi-layer perceptron class.
    This is used via SGD only in the MNIST thing Howon rigged up
    '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)
        self.bp_times = []

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(ourdok_ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(ourdok_ones(self.shape[i]))

        # Build weights matrix (randomly)
        self.weights = []
        for i in range(n-1):
            self.weights.append(ourdok_random(len(self.layers[i]),
                                              len(self.layers[i+1])))

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''
        ''' Data is still in numpy format, hear? '''

        # Set input layer
        for x in xrange(data.size):
            self.layers[0][x] = data[x]

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            # self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))
            self.layers[i] = ourdok_sigmoid(ourdok_vecmatdot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.01, l1=0.001):
        ''' Back propagate error related to target using lrate. '''
        begin_time = time.clock()

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            error = np.dot(deltas[0], self.weights[i].T)
            delta = error*dsigmoid(self.layers[i])
            deltas.insert(0,delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw

        # Return error
        end_time = time.clock()
        self.bp_times.append(end_time - begin_time)
        return (error**2).sum()

    def sparsify(self, thresh=0.01):
        for i in range(len(self.weights)):
            self.weights[i][np.abs(self.weights[i]) < thresh] = 0

def onehots(n):
    arr = np.array([-1.0] * 10)
    arr[n] = 1.0
    return arr

def create_mnist_samples(filename="mnist.pkl.gz"):
    samples = np.zeros(50000, dtype=[('input',  float, 784), ('output', float, 10)])
    with gzip.open(filename, "rb") as f:
        train_set, valid_set, test_set = cPickle.load(f)
        for x in xrange(50000):
            samples[x] = train_set[0][x], onehots(train_set[1][x])
    return samples, 784

def create_cifar_samples(filename="cifar-10-batches-py/data_batch_1"):
    samples = np.zeros(10000, dtype=[('input',  float, 3072), ('output', float, 10)])
    with open(filename, "rb") as f:
        cifar_dict = cPickle.load(f)
        for x in xrange(10000):
            # CIFAR is uint8s, but I would like floats
            samples[x] = cifar_dict["data"][x] / 256.0, onehots(cifar_dict["labels"][x])
    return samples, 3072

def test_network(net, samples):
    correct, total = 0, 0
    for x in xrange(samples.shape[0]):
        total += 1
        in_pat = samples["input"][x]
        out_pat = samples["output"][x]
        out = net.propagate_forward(in_pat)
        if np.argmax(out) == np.argmax(out_pat):
            correct += 1
    # lots of less naive things out there
    return float(correct) / float(total)

def test_conventional_net():
    samples, dims = create_mnist_samples()
    network = MLP(dims, 100, 10)
    for i in xrange(25000):
        if i % 100 == 0:
            print "sample: ", i
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n])
    print test_network(network, samples[40000:40500])
    network.sparsify()
    print test_network(network, samples[40000:40500])
    network.save_weights()

def profile_hidden_range():
    samples, dims = create_mnist_samples()
    networks = [MLP(dims, hids, 10) for hids in range(16, 128)]
    times = []
    for idx, curr_network in enumerate(networks):
        for i in xrange(2000):
            n = np.random.randint(samples.size)
            curr_network.propagate_forward(samples['input'][n])
            curr_network.propagate_backward(samples['output'][n])
        curr_time = np.median(np.array(curr_network.bp_times))
        print idx, curr_time
        times.append(np.log(curr_time))
    plt.plot(times)
    plt.show()

def profile_expando_range():
    samples, dims = create_mnist_samples()
    network = MLP(dims, 16, 10)
    for i in xrange(10000):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n])
    plt.hist(np.abs(network.weights[0].ravel()))
    plt.show()
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    vec = {0: 1, 1: 2, 2: 3}
    mat = {0: {0: 1, 1: 2, 2: 3},
            1: {0: 4, 1: 5, 2: 6},
            2: {0: 7, 1: 8, 2: 9}}
    print ourdok_vecmatdot(vec, mat)
    npvec = np.array([1,2,3])
    npmat = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print npvec
    print npmat
    print np.dot(npvec, npmat)
    # samples, dims = create_mnist_samples()
    # network = MLP(dims, 100, 10)
    # for i in xrange(25000):
    #     if i % 100 == 0:
    #         print "sample: ", i
    #     n = np.random.randint(samples.size)
    #     network.propagate_forward(samples['input'][n])
    #     network.propagate_backward(samples['output'][n])
    # print test_network(network, samples[40000:40500])
    # network.sparsify()
