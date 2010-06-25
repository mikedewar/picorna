import urllib
import json
import time

def google_search(phrase):
        """Get google search results for the phrase"""

        referer = "http://kevinrodrigues.com"

        

        

if __name__ == '__main__':
        google_search('Kevin Rodrigues')

class Genome():
    def __init__(self,name):
        self.name = name
        self.lines = []
    
    def add_line(self,line):
        self.lines.append(line)
        
    def finish(self):
        self.data = "".join(self.lines)
        
    def __str__(self):
        return self.name + "\n" + self.data

class Virus():
    def __init__(self,name):
        self.name = name
        self.genomes = []
        self.label = None
    
    def add_line(self,line):
        if ">" in line:
            if len(self.genomes):
                self.genomes[-1].finish()
            self.genomes.append(Genome(line))
        else:
            self.genomes[-1].add_line(line)
        self.genomes[-1].finish()
    
    def __len__(self):
        return len(self.genomes)
    
    def __getitem__(self,i):
        return self.genomes[i]
    
    def __str__(self):
        return self.name

class Picorna():
    def __init__(self,filename='picornavirus-proteins.fasta'):
        self.viruses = []
        f = open(filename,'r').readlines()
        f = [fi.strip() for fi in f]
        for line in f:
            if "NC_" in line:
                full_name = line.split(",")[0]
                name_elements = full_name.split(' ')
                virus_name = ' '.join(name_elements[1:])
                self.viruses.append(Virus(virus_name))
            else:
                self.viruses[-1].add_line(line)
        self.gen_labels()
    
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
            u = url('"%s"+%s'%(v.name,sample))
            print "query: %s"%u
            return urllib.urlopen(u)
        # this pulls the number of results from the object returned by google
        (key1,key2,key3) = ('responseData','cursor','estimatedResultCount')
        def get_n(result):
            try:
                return int(result[key1][key2][key3])
            except KeyError:
                print result
                return 0
        # these are out labels
        labels = ["insect","mammal","plant"]
        # these are the sample organisms against which to search
        samples = ['bee','rat','strawberry']
        # now go through each virus and choose the label associated with the
        # maximum number of search results
        for i,v in enumerate(self.viruses):
            n = []
            for sample in samples:
                n.append(
                    get_n(json.load(open_url(v,sample)))
                )
                time.sleep(5)
            self.viruses[i].label = labels[n.index(max(n))]
            print "%s is a %s virus"%(v.name,self.viruses[i].label)


if __name__=="__main__":
    v = Picorna()
    print v.viruses[6].genomes[1]
