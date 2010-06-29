import urllib
import json
import time
from mismatch import *

class Protein():
    def __init__(self,name):
        print "initialised %s"%name
        self.name = name
        self.lines = []
    
    def add_line(self,line):
        self.lines.append(line)
        
    def finish(self,k,m):
        self.data = "".join(self.lines)
        print self.data
        self.feature = gen_features(self.data,k,m)
        
    def __str__(self):
        return self.name + "\n" + self.data

class Virus():
    def __init__(self,name,k,m):
        print "initialised %s"%name
        self.name = name
        self.proteins = []
        self.label = None
        self.k = k
        self.m = m
        
    def add_line(self,line):
        if ">" in line:
            if len(self.proteins):
                self.proteins[-1].finish(self.k,self.m)
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
    def __init__(self,k,m,filename='picornavirus-proteins.fasta'):
        self.viruses = []
        self.k = k
        self.m = m
        f = open(filename,'r').readlines()
        f = [fi.strip() for fi in f]
        for line in f:
            print line
            if "NC_" in line:
                full_name = line.split(",")[0]
                name_elements = full_name.split(' ')
                virus_name = ' '.join(name_elements[1:])
                self.viruses.append(Virus(virus_name,self.k,self.m))
            else:
                self.viruses[-1].add_line(line)
        self.viruses[-1].proteins[-1].finish()
    
    def __len__(self):
        return len(self.viruses)
    
    def __getitem__(self,i):
        return self.viruses[i]
    
            
        
    

if __name__=="__main__":
    v = Picorna(k=3,m=2)
    print v.viruses[6].genomes[1]
