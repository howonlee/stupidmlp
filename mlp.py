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
import scipy.sparse as sci_sp
import matplotlib.pyplot as plt
import cPickle
import collections
import time
import sys
import random
import gzip

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

def mat_dsigmoid(mat):
    new_mat = mat.copy()
    new_mat.data = 1.0 - (new_mat.data ** 2)
    return new_mat

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
        self.layers.append(sci_sp.csc_matrix(np.ones(self.shape[0]+1)))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(sci_sp.csc_matrix(np.ones(self.shape[i])))

        # Build weights matrix (randomly)
        self.weights = []
        self.sparsifiers = []
        self.has_sparsified = False
        for i in range(n-1):
            new_weights = (2 * (npr.random((self.layers[i].size, self.layers[i+1].size))) - 1) * 0.00001
            self.weights.append(sci_sp.csc_matrix(new_weights))
            self.sparsifiers.append(np.zeros_like(new_weights))

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''
        ''' Data is still in numpy format, hear? '''

        # Set input layer
        self.layers[0][0, 0:-1] = data
        # for x in xrange(data.size):
        #     self.layers[0][0, x] = data[x]

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(self.layers[i-1].dot(self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.01):
        ''' Back propagate error related to target using lrate. '''
        begin_time = time.clock()

        deltas = []

        # Compute error on output layer
        error = sci_sp.csc_matrix(target - self.layers[-1])
        delta = error.multiply(mat_dsigmoid(self.layers[-1]))
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            error = sci_sp.csc_matrix(deltas[0].dot(self.weights[i].T))
            delta = error.multiply(mat_dsigmoid(self.layers[i]))
            deltas.insert(0,delta)

        # Update weights
        for i in range(len(self.weights)):
            dw = self.layers[i].T.dot(deltas[i])
            self.weights[i] += lrate*dw
            if i < len(self.weights)-1 and self.has_sparsified:
                self.weights[i][self.sparsifiers[i]] = 0
                self.weights[i].eliminate_zeros()

        # Return error
        end_time = time.clock()
        self.bp_times.append(end_time - begin_time)
        return error.sum()

    def expando(self):
        # expand weights by hstack each time
        for i in range(1, len(self.layers)-1):
            print "begin layer: ", self.layers[i].shape
            arr_layer = self.layers[i].toarray()
            self.layers[i] = sci_sp.csc_matrix(np.hstack((arr_layer, arr_layer)))
            print "end layer: ", self.layers[i].shape
        for i in range(0, len(self.weights)-1):
            print "begin weight: ", self.weights[i].shape
            arr_weights = self.weights[i].toarray()
            self.weights[i] = sci_sp.csc_matrix(np.hstack((arr_weights, arr_weights)))
            self.sparsifiers[i] = np.hstack((self.sparsifiers[i], self.sparsifiers[i]))
            print "end weight: ", self.weights[i].shape
        print "begin weight: ", self.weights[-1].shape
        last_weights = self.weights[-1].toarray()
        self.weights[-1] = sci_sp.csc_matrix(np.vstack((last_weights, last_weights)))
        print "end weight: ", self.weights[-1].shape

    def check_sparsity(self):
        for i in range(len(self.weights)):
            print len(self.weights[i].indices), " / ",\
                    reduce(lambda x, y: x * y, self.weights[i].shape)

    def sparsify(self):
        self.has_sparsified = True
        for i in range(len(self.weights)-1): # not the softmax layer
            # so the 50th percentile of existing weights above 0
            # leave it as np.percentile, not median, cuz experimentation
            thresh = np.percentile(
                               np.abs(
                                   self.weights[i].toarray()[np.abs(self.weights[i].toarray()) > 0]
                                ), 50
                              )
            # add to sparsifier
            # kill based upon sparsifier, but in the actual backprop
            self.sparsifiers[i] = np.logical_or(np.abs(self.weights[i].toarray()) < thresh, self.sparsifiers[i])

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
        # needs to be adjusted for neural net's sparsity datastruct
        out = net.propagate_forward(in_pat).toarray().ravel()
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
    samples, dims = create_mnist_samples()
    network = MLP(dims, 8, 10)
    num_epochs = 1
    num_iters = 30000
    prev_time = time.clock()
    for epoch in xrange(num_epochs):
        for i in xrange(num_iters):
            if i % 100 == 0:
                print "==============="
                print "sample: ", i, " / ", num_iters, " time: ", time.clock()
                print "epoch: ", epoch, " time taken: ", time.clock() - prev_time
                prev_time = time.clock()
                network.check_sparsity()
                print "==============="
            n = np.random.randint(samples.size)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n])
        network.expando()
        network.sparsify()
        network.check_sparsity()
    print test_network(network, samples[40000:40500])
