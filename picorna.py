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
    
    def add_line(self,line):
        if ">" in line:
            if len(self.genomes):
                self.genomes[-1].finish()
            self.genomes.append(Genome(line))
        else:
            self.genomes[-1].add_line(line)
        self.genomes[-1].finish()

class Picorna():
    def __init__(self,filename='picornavirus-proteins.fasta'):
        self.viruses = []
        f = open(filename,'r').readlines()
        f = [fi.strip() for fi in f]
        for line in f:
            if "NC_" in line:
                virus_name = Virus(line.split(",")[0])
                self.viruses.append(virus_name)
            else:
                self.viruses[-1].add_line(line)

if __name__=="__main__":
    v = Picorna()
    print v.viruses[6].genomes[1]
