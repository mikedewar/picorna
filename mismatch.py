import numpy as np
import itertools
import timeit

def form_all_kmers(A,k):
    all_kmers = itertools.product(A,repeat=k)
    return np.array([beta for beta in all_kmers])

def gen_features(x,m,beta):
    """
    x : 
        protein sequence
    m : 
        number of mismatches
    beta : array
        all possibly kmers
    """
    k = len(beta[0])
    all_kmers_in_x = [x[i:i+k] for i in range(len(x)-2)]
    alpha = np.array([list(a) for a in all_kmers_in_x])
    # form features
    n1 = alpha.shape[0]
    n2 = beta.shape[0]
    mismatches = np.repeat(alpha, n2, 0) != np.tile(beta, (n1,1))
    tested = np.sum(mismatches,1) <= m
    tested_reshaped = np.reshape(tested,(n2,n1))
    return np.sum(tested_reshaped,1)

    
    