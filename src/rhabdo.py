from mismatch import *
import os
import csv

project_path = '/proj/ar2384/picorna/'

class Rhabdo():
    "class defining a set of Rhabdoviruses"
    def __init__(self,fasta,labels,k,m):
        self.fasta_fh = open(fasta,'r')
        self.labels_fh = open(labels,'r')
        self.k = k
        self.m = m
        self.viruses = []
        cmd = [
            "grep -v NC_ %s"%fasta,
            "grep -v '>'",
            "tr '\n' '!'"
        ]
        x = os.popen(' | '.join(cmd)).next()
        self.beta = form_all_kmers_in_string(self.k,x)
        self.label_dict = {
            1:"plant",
            2:"animal",
        }
    
    def parse_fasta(self):
        for line in self.fasta_fh:
            if line.startswith('>'):
                splitline = line.split()
                virus_id = splitline[0][1:]
                name = " ".join(splitline[1:])
                self.viruses.append(Virus(name,virus_id,self.m,self.beta))
            else:
                self.viruses[-1].add_genome(line.strip())
    
    def parse_labels(self):
        for row in csv.reader(self.labels_fh, delimiter=','):
            name, cls = row
            name_elements = name.split(' ')
            virus_id = name_elements[0]
            try:
                virus = self.get_virus_by_id(virus_id)
            except LookupError:
                print "can't find virus %s with id %s"%(name, virus_id)
            virus.add_label(self.label_dict[int(cls)])
    
    def __len__(self):
        return len(self.viruses)
    
    def __getitem__(self,i):
        return self.viruses[i]
    
    def get_virus_by_id(self,virus_id):
        for v in self.viruses:
            if v.virus_id == virus_id:
                return v
        raise LookupError(id)
             

class Virus():
    "class defining a virus"
    def __init__(self, name, virus_id, m, beta):
        self.name = name
        self.virus_id = virus_id
    def add_genome(self,genome):
        self.genome = genome
    def add_label(self,label):
        self.label = label
                    
    


if __name__ == "__main__":
    fasta = project_path+"data/rhabdovirus-proteins.fasta"
    labels = project_path+"data/rhabdo_classes.csv"
    k = 4
    m = 2
    test = Rhabdo(fasta,labels,k,m)
    test.parse_fasta()
    test.parse_labels()
    
