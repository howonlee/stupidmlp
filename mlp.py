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

def sparse_outer(fst, snd, to_calc):
    new = np.zeros(to_calc.shape)
    fst = fst.toarray()
    snd = snd.toarray()
    for row_idx, col_idx in zip(to_calc.row, to_calc.col):
        # matrices, remember
        new[row_idx, col_idx] = fst[0, row_idx] * snd[0, col_idx]
    return sci_sp.csr_matrix(new)

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
            self.sparsifiers.append(sci_sp.coo_matrix(np.ones_like(new_weights)))

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''
        ''' Data is still in numpy format, hear? '''

        # Set input layer
        self.layers[0][0, 0:-1] = data

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

        # Update weights: this is the bit that scales
        for i in range(len(self.weights)):
            if i < len(self.weights)-1:
                dw = sparse_outer(self.layers[i], deltas[i], self.sparsifiers[i])
            else:
                dw = self.layers[i].T.dot(deltas[i]) ### dense at the last one
            self.weights[i] += lrate*dw

        # Return error
        end_time = time.clock()
        self.bp_times.append(end_time - begin_time)
        return error.sum()

    def check_sparsity(self):
        for i in range(len(self.weights)):
            print >> sys.stderr, len(self.weights[i].indices), " / ",\
                    reduce(lambda x, y: x * y, self.weights[i].shape)

    def random_sparsify(self):
        # experimentation
        self.has_sparsified = True
        for i in range(len(self.weights)-1): # not the softmax layer
            new_sparsifier = self.sparsifiers[i].toarray()
            self.sparsifiers[i] = sci_sp.coo_matrix(np.logical_and(npr.random(size=self.weights[i].shape) > 0.15, new_sparsifier))
            self.weights[i][self.sparsifiers[i]] = 0
            self.weights[i].eliminate_zeros()

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
            new_sparsifier = self.sparsifiers[i].toarray()
            self.sparsifiers[i] = sci_sp.coo_matrix(np.logical_and(np.abs(self.weights[i].toarray()) > thresh, new_sparsifier))
            self.weights[i][np.abs(self.weights[i].toarray()) < thresh] = 0
            self.weights[i].eliminate_zeros()

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

def test_sparsify(num_epochs, num_sparsifications, num_burnin, num_iters, hidden_units):
    total_begin_time = time.clock()
    samples, dims = create_mnist_samples()
    network = MLP(dims, hidden_units, 10)
    for i in xrange(num_burnin):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n])
    # network.stupid_sparsify()
    for x in xrange(num_sparsifications):
        network.sparsify()
    print >> sys.stderr, "burnin finished"
    network.check_sparsity()
    prev_time = time.clock()
    for epoch in xrange(num_epochs):
        for i in xrange(num_iters):
            if i % 100 == 0:
                print >> sys.stderr, "==============="
                print >> sys.stderr, "sample: ", i, " / ", num_iters, " time: ", time.clock()
                print >> sys.stderr, "epoch: ", epoch, " time taken: ", time.clock() - prev_time
                if network.bp_times:
                    print >> sys.stderr, "last bp_time: ", network.bp_times[-1]
                prev_time = time.clock()
                network.check_sparsity()
                print >> sys.stderr, "==============="
            n = np.random.randint(samples.size)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n])
        # was also expanding, but that doesn't work as well
        network.check_sparsity()
    print "num_epochs: ", str(num_epochs), " num_sparsifications: ", str(num_sparsifications), " num_burnin: ", str(num_burnin), " num_iters: ", str(num_iters), " hidden_units: ", str(hidden_units)
    print "accuracy: ", test_network(network, samples[40000:40500])
    print "total time: ", str(time.clock() - total_begin_time)

if __name__ == '__main__':
    for x in [2,4,8,16,32,64]:
        test_sparsify(num_epochs=1, num_sparsifications=0, num_burnin=200, num_iters=30000, hidden_units=x)
    print "now sparsifying..."
    for x in [0,1,2,3,4,5]:
        test_sparsify(num_epochs=1, num_sparsifications=x, num_burnin=200, num_iters=30000, hidden_units=128)
    for x, y in zip([0,1,2,3,4,5], [4,8,16,32,64,128]):
        test_sparsify(num_epochs=1, num_sparsifications=x, num_burnin=200, num_iters=30000, hidden_units=y)
