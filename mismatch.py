import numpy as np
import itertools
import timeit

def form_all_kmers(A,k):
    """
    A : list
        alphabet - all possible characters
    k : int
        the length of subsequences you're after
    
    Returns
    -------
    beta : array
        all possible kmers that can be formed by the alphabet A
    """
    all_kmers = itertools.product(A,repeat=k)
    return np.array([beta for beta in all_kmers])

def form_all_kmers_in_string(k,x):
    """
    k : int 
        the length of the subseqeunces you're after
    x : string
        the string from which you'd like to form all kmers
    """
    strings = np.empty((k,len(x)-k),dtype=str)
    x = list(x)
    for i in range(k):
        strings[i,:] = x[i:-(k-i)]
    # this is all the kmers
    return np.unique([''.join(kmer) for kmer in strings.T if '!' not in kmer])

def gen_features(x,m,beta):
    """
    a feature of `x` is the count in `x` of each kmer in `beta`, where the 
    kmers in `x` are allowed to mismatch each element of beta by `m` 
    mismatches.
    
    Arguments
    ---------
    
    x : list
        protein sequence
    m : int
        number of mismatches
    beta : array
        all possible kmers
            
    Returns
    -------
    features : array
        count in `x` of each kmer in `beta` varying by `m` mismatches.
    """
    k = len(beta[0])
    all_kmers_in_x = form_all_kmers_in_string(k, x)
    alpha = np.array([list(a) for a in all_kmers_in_x])
    beta = np.array([list(b) for b in beta])
    # form features
    n1,n2 = alpha.shape[0], beta.shape[0]
    mismatches = np.repeat(alpha, n2, 0) != np.tile(beta, (n1,1))        
    tested = np.sum(mismatches,1) <= m
    tested_reshaped = np.reshape(tested,(n2,n1))
    return np.sum(tested_reshaped,1)

    
    