import scipy.sparse as sci_sp
import numpy as np
import numpy.random as npr
import random
import time

### http://stackoverflow.com/questions/18595981/improving-performance-of-multiplication-of-scipy-sparse-matrices wtf

def sparse_col_vec_dot(csc_mat, csc_vec):
    # row numbers of vector non-zero entries
    v_rows = csc_vec.indices
    v_data = csc_vec.data
    # matrix description arrays
    m_dat = csc_mat.data
    m_ind = csc_mat.indices
    m_ptr = csc_mat.indptr
    # output arrays
    sizes = m_ptr.take(v_rows+1) - m_ptr.take(v_rows)
    sizes = np.concatenate(([0], np.cumsum(sizes)))
    data = np.empty((sizes[-1],), dtype=csc_mat.dtype)
    indices = np.empty((sizes[-1],), dtype=np.intp)
    indptr = np.zeros((2,), dtype=np.intp)

    for j in range(len(sizes)-1):
        slice_ = slice(*m_ptr[[v_rows[j] ,v_rows[j]+1]])
        np.multiply(m_dat[slice_], v_data[j], out=data[sizes[j]:sizes[j+1]])
        indices[sizes[j]:sizes[j+1]] = m_ind[slice_]
    indptr[-1] = len(data)
    ret = sci_sp.csc_matrix((data, indices, indptr),
                         shape=csc_vec.shape)
    ret.sum_duplicates()

    return ret

def sparsify_mat(arr, thresh=0.9999):
    # there are only improvements at... very sparse patts
    new_arr = arr.copy()
    for x in xrange(arr.shape[0]):
        for y in xrange(arr.shape[1]):
            if random.random() < thresh:
                new_arr[x,y] = 0
    return new_arr

def sparsify_vec(vec, thresh=0.99):
    new_arr = vec.copy()
    for x in xrange(vec.shape[0]):
        if random.random() < thresh:
            new_arr[x] = 0
    return new_arr

def sparse_vs_dense():
    # C++ versus ATLAS, C++ loses
    for arr_size in [50, 100, 200, 400, 800, 1600]:
        # rand_vec = np.matrix(sparsify_vec(npr.random(arr_size)))
        rand_vec = np.matrix(npr.random(arr_size))
        csr_vec = sci_sp.csr_matrix(rand_vec)
        rand_mat = np.matrix(sparsify_mat(npr.random((arr_size, arr_size))))
        csr_mat = sci_sp.csr_matrix(rand_mat)
        print "array size: ", arr_size
        curr_sparses, curr_denses = [], []
        for x in xrange(1000):
            dense_begin_time = time.clock()
            dense_res = rand_mat * rand_vec.T
            dense_end_time = time.clock()
            curr_denses.append(dense_end_time - dense_begin_time)
        for x in xrange(1000):
            sparse_begin_time = time.clock()
            sparse_res = sparse_col_vec_dot(csr_mat, csr_vec.T)
            sparse_end_time = time.clock()
            curr_sparses.append(sparse_end_time - sparse_begin_time)
        print np.median(np.array(curr_sparses)), np.median(np.array(curr_denses))

def compare_sparses():
    threshes = map(lambda x: float(x) / 10, range(10))
    for thresh in threshes:
        for x in xrange(100):
            vec = sci_sp.csr_matrix(np.matrix(sparsify_vec(npr.random(100), thresh=thresh)))
            mat = sci_sp.csr_matrix(np.matrix(sparsify_mat(npr.random((100, 100)), thresh=thresh)))
            begin_time = time.clock()
            res = sparse_col_vec_dot(mat, vec.T)
            end_time = time.clock()
            print "thresh: ", thresh, " , time: ", end_time - begin_time


if __name__ == "__main__":
    compare_sparses()
