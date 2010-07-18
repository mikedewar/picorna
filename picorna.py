import urllib
import json
import time
from mismatch import *
import csv


class Protein():
    """
    Class describing a protein in terms of its amino acid sequence
    """
    def __init__(self,name):
        """
        name : string
            name of the protein
        """
        print "\tinitialised %s"%name
        self.name = name
        self.lines = []
        self.label = None
    
    def add_line(self,line):
        """
        this adds a line of symbols to the (temporary) lines variable
        
        line : string
            a line of protein symbols from a fasta file
        """
        self.lines.append(line)
        
    def finish(self,m,beta):
        """
        this finishes off the parsing of a single protein from a fasta file.
        It also generates the feature set - the count of each possible kmer in
        the protein that are within m mismatches.
        
        m : int
            number of mismatches allowed
        beta : list
            all possible kmers
        """
        print "\tfinishing %s"%self.name
        self.data = "".join(self.lines)
        self.feature = gen_features(self.data,m,beta)
        
    def __str__(self):
        return self.name + "\n" + self.data

class Virus():
    """
    class describing a virus as a collection of proteins
    """
    def __init__(self,name,virus_id,m,beta):
        """
        Arguments
        ---------
        name : string
            name of the virus
        virus_id : string
            unique id of the virus
        m : int
            number of allowed mismatches
        beta : list
            all possible kmers
        
        Notes
        -----
        The arguments m and beta are for generating features. See 
        protein.finish() for more info.
        """
        print "initialised %s with id %s"%(name,virus_id)
        self.name = name
        self.id = virus_id
        self.proteins = []
        self.label = None
        self.m = m
        self.beta = beta
        
    def add_line(self,line):
        """
        adds a line from a fasta file to the virus definition. This either
        starts a new protein or adds a line to the current protein.
        """
        if ">" in line:
            if len(self.proteins):
                self.proteins[-1].finish(self.m,self.beta)
            self.proteins.append(Protein(line))
        else:
            self.proteins[-1].add_line(line)
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self,i):
        return self.proteins[i]
    
    def __str__(self):
        return self.name

class Picorna():
    """
    class describing a set of picorna viruses
    """
    def __init__(self,k,m):
        """
        k : int
            length of kmers to consider
        m : int
            largest number of mismatches
        """
        self.k = k
        self.m = m
        self.A = list('ACDEFGHIKLMNPQRSTVWY')
        self.beta = form_all_kmers(self.A,self.k)
        self.viruses = []
        # a dictionary of label numbers to labels
        self.label_dict = {
            1:"invertebrate",
            2:"plant",
            3:"vertebrate"
        }
        
    
    def parse(self,filename='picornavirus-proteins.fasta',max_v=None):
        """
        This method parses a fasta file, populating the objects as it goes.
        
        Arguments
        ---------
        
        filename : string
            filename of a multi-virus, multi-protein fasta file
        max_v : int
            maximum number of viruses you want - used for debugging
        """
        f = open(filename,'r').readlines()
        f = [fi.strip() for fi in f]
        for line in f:
            if "NC_" in line:
                full_name = line.split(",")[0]
                name_elements = full_name.split(' ')
                virus_name = ' '.join(name_elements[1:])
                virus_id = name_elements[0]
                self.finish_last_protein()
                if max_v:
                    if len(self.viruses) > max_v:
                        break
                self.viruses.append(Virus(virus_name,virus_id,self.m,self.beta))
            else:
                self.viruses[-1].add_line(line)
        self.finish_last_protein()
        self.assign_classes()
    
    def finish_last_protein(self):
        """
        this is called at the very end of the parsing to finish off the last
        protein.
        """
        if len(self.viruses):
            if len(self.viruses[-1].proteins):
                self.viruses[-1].proteins[-1].finish(self.m,self.beta)
    
    def assign_classes(self,classfile="classes.csv"):
        """
        This class reads the classfile which contains the ids, names and class
        labels, and associates the appropriate label with each virus and
        protein stored in the Picorna object.
        """
        
        for row in csv.reader(open(classfile,'r'), delimiter=','):
            try:
                name, cls = row
            except:
                print row
                raise
            name_elements = name.split(' ')
            virus_id = name_elements[0]
            try:
                virus = self.get_virus_by_id(virus_id)
            except LookupError:
                print "can't find virus %s with id %s"%(name, virus_id)
            virus.label = self.label_dict[int(cls)]
            for protein in virus.proteins:
                protein.label = self.label_dict[int(cls)]
    
    def __len__(self):
        return len(self.viruses)
    
    def __getitem__(self,i):
        return self.viruses[i]
    
    def get_virus_by_id(self,id):
        for v in self.viruses:
            if v.id == id:
                return v
        raise LookupError(id)
    
    def summarise(self):
        """
        This method collects together all the feature and class label 
        information in the Picorna object and creates a data matrix and a 
        class matrix
        
        Returns
        -------
        X : DxN array
            where D = number of kmers, N = number of proteins and the array 
            elements are the kmer counts
        Y : KxN array
            where K = number of classes and Yij = 1 if the jth protein belongs
            to the ith class, otherwise Yij = -1)
        """
        feature_list = []
        for virus in self:
            for protein in virus:
                feature_list.append(protein.feature)
        X = np.vstack(feature_list).T
        
        Y = np.empty((len(self.label_dict), X.shape[1]))
        for i in range(Y.shape[0]):
            j = 0
            for virus in self:
                for protein in virus:
                    if protein.label == self.label_dict[i+1]:
                        Y[i,j] = 1
                    else:
                        Y[i,j] = -1
                    j += 1
        return X, Y


if __name__=="__main__":
    import csv
    
    v = Picorna(k=3,m=2)
    v.parse(max_v = 2)
    
    X,Y = v.summarise()
            

    
    
    
    
