import urllib
import json
import time
from mismatch import *

class Protein():
    def __init__(self,name):
        print "\tinitialised %s"%name
        self.name = name
        self.lines = []
    
    def add_line(self,line):
        self.lines.append(line)
        
    def finish(self,m,beta):
        print "\tfinishing %s"%self.name
        self.data = "".join(self.lines)
        self.feature = gen_features(self.data,m,beta)
        
    def __str__(self):
        return self.name + "\n" + self.data

class Virus():
    def __init__(self,name,m,beta):
        print "initialised %s"%name
        self.name = name
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
    
    def parse(self,filename='picornavirus-proteins.fasta'):
        """
        filename : string
            filename of a multi-virus, multi-protein fasta file
        """
        f = open(filename,'r').readlines()
        f = [fi.strip() for fi in f]
        for line in f:
            if "NC_" in line:
                full_name = line.split(",")[0]
                name_elements = full_name.split(' ')
                virus_name = ' '.join(name_elements[1:])
                self.finish_last_protein()
                self.viruses.append(Virus(virus_name,self.m,self.beta))
            else:
                self.viruses[-1].add_line(line)
        self.finish_last_protein()
    
    def finish_last_protein(self):
        if len(self.viruses):
            if len(self.viruses[-1].proteins):
                self.viruses[-1].proteins[-1].finish(self.m,self.beta)
    
    def __len__(self):
        return len(self.viruses)
    
    def __getitem__(self,i):
        return self.viruses[i]
    
    

if __name__=="__main__":
    v = Picorna(k=3,m=2)
    v.parse()
