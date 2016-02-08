import scipy.sparse as sci_sp
import numpy as np
import numpy.random as npr
import random
import time

def sparsify(arr, thresh=0.5):
    new_arr = arr.copy()
    for x in xrange(arr.shape[0]):
        for y in xrange(arr.shape[1]):
            if random.random() < thresh:
                new_arr[x,y] = 0
    return new_arr

if __name__ == "__main__":
    for arr_size in [50, 100, 200, 400, 800, 1600]:
        rand_vec = np.matrix(npr.random(arr_size))
        csc_vec = sci_sp.csc_matrix(rand_vec)
        rand_mat = np.matrix(sparsify(npr.random((arr_size, arr_size))))
        csc_mat = sci_sp.csc_matrix(rand_mat)
        print "array size: ", arr_size
        for x in xrange(10):
            dense_begin_time = time.clock()
            dense_res = rand_vec * rand_mat
            dense_end_time = time.clock()
            print "dense time: ", dense_end_time - dense_begin_time
        for x in xrange(10):
            sparse_begin_time = time.clock()
            sparse_res = csc_vec * csc_mat
            sparse_end_time = time.clock()
            print "sparse time: ", sparse_end_time - sparse_begin_time
