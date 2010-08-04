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
        number of allowed mismatches
    beta : array
        all possible kmers
            
    Returns
    -------
    features : array
        count in `x` of each kmer in `beta` varying by `m` mismatches.
    """
    k = len(beta[0])
    y = np.array([list(yi) for yi in form_all_kmers_in_string(k, x)])
    b = np.array([list(bi) for bi in beta])    
    
    count = np.sum(np.reshape(
            np.sum(np.repeat(b,len(y),0) != np.tile(y,[len(b),1]), 1) <= m, 
            (len(beta),len(y))
        ),
        1
    )
    assert len(count) == len(beta)
    return count
    
    
    
