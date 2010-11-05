from mismatch import *
import os
import csv


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
            2:"vertebrate"
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
    
    def summarise(self):
        """
        This method collects together all the feature and class label 
        information in the Rhabdo object and creates a data matrix and a 
        class matrix
        
        Returns
        -------
        X : DxN array
            where D = number of kmers, N = number of viruses and the array 
            elements are the kmer counts within the mismatch value
        Y : KxN array
            where K = number of classes and Yij = 1 if the jth virus belongs
            to the ith class, otherwise Yij = -1
        D : dict
            a mapping from the row index of X to each of the D kmers
        """
        X = []
        for mi in range(self.m):
            feature_list = []
            for virus in self:
                feature_list.append(virus.feature[:,mi])
            X.append(np.vstack(feature_list).T)
        
        Y = np.empty((len(self.label_dict), X[0].shape[1]))
        for i in range(Y.shape[0]):
            j = 0
            for virus in self:
                if virus.label == self.label_dict[i+1]:
                    Y[i,j] = 1
                else:
                    Y[i,j] = -1
                j += 1
        return X, Y, dict(zip(range(len(self.beta)), self.beta))
             

class Virus():
    "class defining a virus"
    def __init__(self, name, virus_id, m, beta):
        self.name = name
        self.virus_id = virus_id
        self.m = m
        self.beta = beta
    def add_genome(self,genome):
        self.genome = genome
        self.feature = gen_features(self.genome,self.m,self.beta)
    def add_label(self,label):
        self.label = label
                    
    


if __name__ == "__main__":
    fasta = "../data/rhabdovirus-proteins.fasta"
    labels = "../data/rhabdo_classes.csv"
    k = 4
    m = 2
    test = Rhabdo(fasta,labels,k,m)
    test.parse_fasta()
    test.parse_labels()
    
    X,Y,d = test.summarise()
    
    print X
    print "\n"
    print Y
    print "\n"
    print d
    
    
    