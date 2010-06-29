import numpy as np
import itertools

class Neighbourhood():
    
    def __init__(self,k,m):
        """
        m : int
            number of allowed mismatches
        """
        self.m = m
    
    def __call__(self,alpha):
        return NeighbourhoodTester(alpha,self.m)
    

class NeighbourhoodTester():
    
    def __init__(self,alpha,m):
        self.alpha = alpha
        self.m = m
        
    def __contains__(self,beta):
        assert len(self.alpha) == len(beta)
        return  self.m >= sum([a!=b for (a,b) in zip(self.alpha,beta)])
    

def gen_features(x,k,m,A=None):
    """
    x : 
        protein sequence
    k : 
        length of substring
    m : 
        number of mismatches
    A : list
        alphabet (defaults to the unique values in x)
    """
    if not A:
        A = np.unique(x)
    all_kmers_in_x = np.unique([
        kmer_i for kmer_i in itertools.ifilter(
            lambda s: s in x, 
            [''.join(a) for a in itertools.combinations(x,k)]
        )
    ])
    all_kmers = [''.join(a) for a in itertools.product(A,repeat=k)]
    # form features
    N_km = Neighbourhood(k,m)
    phi = lambda alpha,beta: (0,1)[beta in N_km(alpha)]
    Phi = lambda alpha: [phi(alpha,beta) for beta in all_kmers]
    return np.sum(np.array([Phi(alpha) for alpha in all_kmers_in_x]),0)

if __name__ == "__main__":
    # this is a line from a protein
    x = "MEHTPLLSSFPITLLDHCGGNRKIHWTRCYEQKYWLPFSCCKVASRLPCVNVRRSYMWYSKKQSKWIYLS"
    print gen_features(x,k=3,m=2)
    
    
    