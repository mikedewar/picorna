import numpy as np
import itertools
import timeit
import time
import pdb

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
    strings = np.empty((k, len(x)-k), dtype=str)
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
    
    print "beta contains %s kmers"%len(beta)
    print "the current string contains %s kmers"%y.shape[0]

    starttime = time.time()
    count = np.zeros((len(beta),m),dtype=np.int16)
    for i,yi in enumerate(y):
        num_mismatches = np.sum(b != yi,1)
        for mi in range(m):
            count[:,mi] += (num_mismatches<=mi)
    print "Loop Time = %.4f" % (time.time() - starttime)

    """
    
    num_chunks = 10
    def chunk(y):
        chunk_length = round(len(y)/num_chunks)
        for i in range(num_chunks):
            yield y[i*chunk_length:(i+1)*chunk_length]

    count = np.zeros(len(beta),dtype=np.int16)
    for i,yi in enumerate(chunk(y)):
        print "processing chunk %s of %s"%(i+1,num_chunks)
        count += np.sum( 
            np.reshape( 
                np.sum( 
                    np.repeat(          # repeat is len(yi)*len(b) x k
                        b ,             # b is the array of all kmers
                        len(yi),        # numer of repeats
                        0               # repeat dim (rows)
                    ) 
                    != 
                    np.tile(            # tile is len(yi)*len(b) x k
                        yi,             # yi is the ith chunk of all kmers in string
                        [len(b),1]      # tile shape
                    ), 
                    1                   # inner sum: result is len(yi)*len(b) x 1
                ) <= m, 
                (len(beta), len(yi))    # reshape size
            ),
            1                           # outer sum: result is len(beta)
        )
    assert len(count) == len(beta)
    """

    return count
    
