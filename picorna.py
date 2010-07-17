import urllib
import json
import time
from mismatch import *
import csv


class Protein():
    def __init__(self,name):
        print "\tinitialised %s"%name
        self.name = name
        self.lines = []
        self.label = None
    
    def add_line(self,line):
        self.lines.append(line)
        
    def finish(self,m,beta):
        print "\tfinishing %s"%self.name
        self.data = "".join(self.lines)
        self.feature = gen_features(self.data,m,beta)
        
    def __str__(self):
        return self.name + "\n" + self.data

class Virus():
    def __init__(self,name,virus_id,m,beta):
        print "initialised %s with id %s"%(name,virus_id)
        self.name = name
        self.id = virus_id
        self.proteins = []
        self.label = None
        self.m = m
        self.beta = beta
        
    def add_line(self,line):
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
        if len(self.viruses):
            if len(self.viruses[-1].proteins):
                self.viruses[-1].proteins[-1].finish(self.m,self.beta)
    
    def assign_classes(self,classfile="classes.csv"):
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
        # X : a DxN matrix (where D = number of kmers, N = number of proteins and 
        # the matrix elements are the kmer counts)
        feature_list = []
        for virus in self:
            for protein in virus:
                feature_list.append(protein.feature)
        X = np.vstack(feature_list).T
    
        # Y : a KxN matrix (where K = number of classes and Yij = 1 if
        # the jth protein belongs to the ith class, otherwise Yij = -1)
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
            

    
    
    
    
