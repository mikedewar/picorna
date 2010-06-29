import urllib
import json
import time
from mismatch import *

class Protein():
    def __init__(self,name):
        self.name = name
        self.lines = []
    
    def add_line(self,line):
        self.lines.append(line)
        
    def finish(self,k,m):
        self.data = "".join(self.lines)
        self.feature = gen_features(self.data,k,m)
        
    def __str__(self):
        return self.name + "\n" + self.data

class Virus():
    def __init__(self,name,k,m):
        self.name = name
        self.proteins = []
        self.label = None
        self.k = k
        self.m = m
        
    def add_line(self,line):
        if ">" in line:
            if len(self.proteins):
                self.proteins[-1].finish()
            self.proteins.append(Protein(line))
        else:
            self.proteins[-1].add_line(line)
        self.proteins[-1].finish(self.k,self.m)
    
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
            if "NC_" in line:
                full_name = line.split(",")[0]
                name_elements = full_name.split(' ')
                virus_name = ' '.join(name_elements[1:])
                print "processing %s"%virus_name
                self.viruses.append(Virus(virus_name,self.k,self.m))
            else:
                self.viruses[-1].add_line(line)
        #self.gen_labels()
    
    def __len__(self):
        return len(self.viruses)
    
    def __getitem__(self,i):
        return self.viruses[i]
    
    def gen_labels(self):
        url_base = "http://ajax.googleapis.com/ajax/services/search/web?v=1.0&"
        # this forms a url
        url = lambda phrase: url_base + urllib.urlencode({'q':phrase})
        # this returns the file associated with a url + query
        def open_url(v,sample):
            name = v.name.split(" ")[:3]
            u = url('"%s"+%s'%(name,sample))
            print "query: %s"%u
            return urllib.urlopen(u)
        # this pulls the number of results from the object returned by google
        (key1,key2,key3) = ('responseData','cursor','estimatedResultCount')
        def get_n(result):
            try:
                return int(result[key1][key2][key3])
            except KeyError:
                return 0
        # these are out labels
        labels = ["invertebrate"] + ["vertebrate"] + ["plant"]
        # these are the sample organisms against which to search
        samples = ['bee','rat','strawberry']
        # now go through each virus and choose the label associated with the
        # maximum number of search results
        for i,v in enumerate(self.viruses):
            n = []
            for sample in labels:
                n.append(
                    get_n(json.load(open_url(v,sample)))
                )
                time.sleep(1)
            if sum(n):
                self.viruses[i].label = labels[n.index(max(n))]
                print "%s is a %s virus"%(v.name,self.viruses[i].label)
                print "insect: %s, mammal: %s, plant:%s"%tuple(n)
            else:
                self.viruses[i].label = "undetermined"
                print "%s is undetermined"%v.name
            
        
    

if __name__=="__main__":
    v = Picorna(k=3,m=2)
    print v.viruses[6].genomes[1]
