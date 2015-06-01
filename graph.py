from numpy import ones, zeros, matrix, squeeze, asarray, float16, sum, savez
from numpy.linalg import det
from scipy.linalg import eig
from scipy.sparse import dok_matrix, lil_matrix, coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.io import mmread, mmwrite
from math import sqrt, floor
from itertools import combinations
from gc import collect

from pysparse import spmatrix

from thread import allocate_lock

from time import time


class Graph:

    def __init__(self, n, edges=[]):
        self.W = spmatrix.ll_mat_sym(n)
        self.n = n
        self.file_tmp = 'matrix_out'
        for edge in edges:
            self.add_edge(edge[0], edge[1], edge[2])


    def add_edge(self, ver1, ver2, weight):
        self.W[ver1, ver2] = weight


    def set_diag(self, values, k=0):
        self.W.put(values, range(self.n), range(self.n))


    def ready(self):
        pass


    def get_edge(self, ver1, ver2):
        return self.W[ver1, ver2]


    def second_smallest(self, a_list):
        m = min(a_list)
        ret = min([l for l in a_list if l != m])
        return (ret, list(a_list).index(ret))

    #@profile
    def reduce_to_eig_problem(self):
        n = self.n
        range_n = range(n)

        self.W.export_mtx(self.file_tmp)
        self.W.delete_rowcols(zeros(n, dtype=int))
        self.W.compress()
        del self.W
        self.W = 0
        collect()
        self.W = csr_matrix(mmread(self.file_tmp))

        # D is a diagonal matrix with sum_j(W[i, j]) at ith diag element
        data = zeros(n, dtype=float16)
        for i in range_n:
            data[i] = self.W.getrow(i).sum()
        D = csr_matrix((data, (range_n, range_n)), shape=(n, n), dtype=float16)

        # D^(-1/2)
        data2 = zeros(n, dtype=float16)
        for i in range_n:
            data2[i] = 1 / sqrt(data[i] + 1E-4)
        D_minus_1_2 = csr_matrix((data2, (range_n, range_n)), shape=(n, n), dtype=float16)

        A = D_minus_1_2 * (D - self.W) * D_minus_1_2
        return A



    def lanczos_optimal_cut(self):
        A = self.reduce_to_eig_problem()

        values, vectors = eigsh(A, k=min(6, A.shape[0] - 1), which='LM', sigma=1E-5)

        minim, minim_idx = self.second_smallest(values)
        result = squeeze(asarray(vectors[:,minim_idx]))

        return (set([i for i in range(len(result)) if result[i] >= 0]),
                set([i for i in range(len(result)) if result[i] < 0]))


    def calculate_cut(self, set1, set2):
        cut = 0
        for v in set1:
            for w in set2:
                cut += self.get_edge(v, w)
        return cut


    def calculate_normalized_cut(self, set1, set2):
        ncut1 = 1. * sum([self.get_edge(v, w) for v in set1 for w in range(self.n)])
        ncut2 = 1. * sum([self.get_edge(v, w) for v in set2 for w in range(self.n)])
        cut = self.calculate_cut(set1, set2)
        return (cut / ncut1) + (cut / ncut2)


    def generate_all_partitions(self):
        _all = set(range(self.n))
        for i in range(int(floor((1. * self.n) / 2.))):
            for l in combinations(range(self.n), i+1):
                yield set(l), set(_all) - set(l)


    def __str__(self):
        return self.W.__str__()
