import numpy as np
import cPickle
import matplotlib.pyplot as plot
import pdb
import sys

# compile hit matrix
def compile_hit_matrix(proteins,kmers,m):
    col_size = 100
    N_proteins = len(proteins)
    N_kmers = len(kmers)
    hit_matrix = np.zeros((N_proteins,col_size+1,N_kmers),dtype='float')
    kmer_length = len(kmers[0])

    options = dict()
    for i in range(m+1):
        options[i] = 1.

    pidx = 0
    for protein in proteins:
        hit_matrix[pidx,0,:] = protein[1]
        protein = protein[0]
        protein_length = len(protein)
        for c in range(protein_length-kmer_length+1):
            for kmer in kmers:
                kidx = kmers.index(kmer)
                mismatch = (np.array(list(protein[c:c+kmer_length]))!=np.array(list(kmer))).sum()
                try:
                    value = options[mismatch]
                    left_col = int(c * float(col_size) / (protein_length-kmer_length+1)) + 1
                    right_col = max([int((c+kmer_length) * float(col_size) / (protein_length-kmer_length+1)),left_col+1])
                    hit_value = hit_matrix[pidx,left_col:right_col,kidx]
                    hit_matrix[pidx,left_col:right_col,kidx] = hit_value * (hit_value>=value) + value * (hit_value<value)
                except KeyError:
                    continue
        pidx += 1

    return hit_matrix

# plot hit matrix
def plot_hit_matrix(hit_matrix,k,m,fold,kmers,viruses):
    color_scheme = {
        'red'   :   np.array([255,0,0]).reshape(1,3)/255.,
        'green' :   np.array([7,255,19]).reshape(1,3)/255.,
        'blue'  :   np.array([12,112,232]).reshape(1,3)/255.,
    'lightblue' :   np.array([0,238,255]).reshape(1,3)/255.,
        'purple':   np.array([152,78,163]).reshape(1,3)/255.,
        'brown' :   np.array([255,127,0]).reshape(1,3)/255.,
        'yellow':   np.array([232,229,34]).reshape(1,3)/255.,
        'white' :   np.array([255,255,255]).reshape(1,3)/255.,
        'black' :   np.array([0,0,0]).reshape(1,3)/255.,
        'grey1' :   np.array([38,38,37]).reshape(1,3)/255.,
        'grey2' :   np.array([18,18,17]).reshape(1,3)/255.,
    }

    colors = ['red','green','blue','purple','brown','lightblue','yellow']
    (R,C,ig) = hit_matrix.shape
    C = C-1
    V = 7
    data = np.zeros((R,C,3),dtype='float')
    for i in range(V):
        data += hit_matrix[:,1:,i:i+1]*color_scheme[colors[i]]
    
    idx = (hit_matrix[:,0,0]==1).nonzero()[0]
    data[idx,:,:] = data[idx,:,:] + (1-(data[idx,:,:].sum(2)>0)).reshape(idx.size,C,1)*color_scheme['grey2']
    idx = (hit_matrix[:,0,0]==2).nonzero()[0]
    data[idx,:,:] = data[idx,:,:] + (1-(data[idx,:,:].sum(2)>0)).reshape(idx.size,C,1)*color_scheme['grey1']
    idx = (hit_matrix[:,0,0]==3).nonzero()[0]
    data[idx,:,:] = data[idx,:,:] + (1-(data[idx,:,:].sum(2)>0)).reshape(idx.size,C,1)*color_scheme['grey2']

    fig = plot.figure()
    im = fig.add_subplot(111)
    im.imshow(data,aspect=0.3,interpolation='nearest')
    im.set_xticks([0,data.shape[1]-1])
    im.set_xlabel('Relative location')
    y_labels = ('Invertebrate','Plant','Vertebrate')
    y_label_loc = []
    for c in np.unique(hit_matrix[:,0,0]):
        y_label_loc.append(-1.*int(np.mean((hit_matrix[:,0,0]==i).nonzero()[0])))
    im.set_yticks([])
    im.set_ylabel('Class')
#    plot.yticks(range(hit_matrix.shape[0]), tuple(viruses), rotation=50, fontsize=3)
    im.set_title('k = %d, m = %d, fold = %d' % (k,m,fold))

    fname = '../Adaboost/kmer_visualization_%d_%d_%d.pdf' % (k,m,fold)
    fig.savefig(fname,dpi=(100),format='pdf')


if __name__=="__main__":
    # k,m values
    k = int(sys.argv[1])
    m = int(sys.argv[2])
    fold = int(sys.argv[3])
    
    # load classes
    classes = dict()
    c = open('../data/picorna_classes.csv','r')
    for line in c:
        row = line.strip().split(',')
        virus_name = ' '.join(row[0].split()[1:])
        classes[row[0].split()[0]] = [virus_name,int(row[1])]

    # load kmers
    kmers = []
    f = open('../Adaboost/errortree_%d_%d_%d.txt' % (k,m,fold),'r')
    f.readline()
    for l in range(7):
        line = f.readline()
        kmers.append(line.strip().split()[1])
    f.close()

    # load protein strings
    proteins = []
    p = open('../data/picornavirus-proteins.fasta','r')
    protein = 'A'
    label = 0
    viruses = []
    for line in p:
        if 'NC_' in line or '>' in line:
            proteins.append([protein,label])
            if 'NC_' in line:
                row = line.strip().split(',')
                virus_name = ' '.join(row[0].split()[1:])
                virus_id = row[0].split()[0]
                viruses.append(virus_id)
                label = classes[virus_id][1]
                protein = ''
            else:
                viruses.append(virus_id)
                protein = ''
                continue 
        else:
            protein += line.strip()
    c.close()
    p.close()
    proteins.pop(0)
    
    hit_matrix = compile_hit_matrix(proteins,kmers,m)

    # save compiled data
    f = open('../Adaboost/hitmatrix_%d_%d_%d.pkl' % (k,m,fold),'w')
    cPickle.Pickler(f,protocol=2).dump(hit_matrix)
    cPickle.Pickler(f,protocol=2).dump(viruses)
    cPickle.Pickler(f,protocol=2).dump(classes)
    f.close()
    """

    # load data
    f = open('../Adaboost/hitmatrix_%d_%d_%d.pkl' % (k,m,fold),'r')
    hit_matrix = cPickle.load(f)
    viruses = cPickle.load(f)
    classes = cPickle.load(f)
    f.close()
    """
    
    sort_indices = hit_matrix[:,0,0].argsort()
    sort_virus_id = [viruses[i] for i in sort_indices]
    sort_viruses = [classes[v][0] for v in sort_virus_id]
    plot_hit_matrix(hit_matrix[sort_indices,:,:],k,m,fold,kmers,sort_viruses)