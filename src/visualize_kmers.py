import numpy as np
import cPickle
import matplotlib.pyplot as plot
from matplotlib.transforms import Bbox
from matplotlib.colors import colorConverter as convert
import pdb
import sys

# compile hit matrix
def compile_hit_matrix(proteins,kmers,m):
    col_size = 500
    N_proteins = len(proteins)
    N_kmers = len(kmers)
    hit_matrix = np.zeros((N_proteins,col_size+1,N_kmers),dtype='float')
    kmer_length = len(kmers[0])

    # this dictionary stores a set of alpha values
    # (transparency) for each mismatch.
    options = dict()
    for i in range(m+1):
        options[i] = 1.

    for pidx, protein in enumerate(proteins):
        # first column stores the virus class
        hit_matrix[pidx,0,:] = protein[1]
        protein = protein[0]
        protein_length = len(protein)
        for c in range(protein_length-kmer_length+1):
            for kidx, kmer in enumerate(kmers):
                mismatch = (np.array(list(protein[c:c+kmer_length]))!=np.array(list(kmer))).sum()
                try:
                    value = options[mismatch]
                    left_col = int(c * float(col_size) / (protein_length-kmer_length+1)) + 1
                    right_col = max([int((c+kmer_length) * float(col_size) / (protein_length-kmer_length+1)),left_col+1])
                    hit_value = hit_matrix[pidx,left_col:right_col,kidx]
                    hit_matrix[pidx,left_col:right_col,kidx] = hit_value * (hit_value>=value) + value * (hit_value<value)
                except KeyError:
                    continue

    return hit_matrix

# plot hit matrix
def plot_hit_matrix(hit_matrix,k,m,fold,kmers,virus_family,project_path):
    background_colors = {
        'white' :   np.array([255,255,255]).reshape(1,3)/255.,
        'black' :   np.array([0,0,0]).reshape(1,3)/255.,
         'grey' :   np.array([38,38,38]).reshape(1,3)/255.,
     'darkgrey' :   np.array([18,18,18]).reshape(1,3)/255.,
     'offwhite' :   np.array([235,235,235]).reshape(1,3)/255.,
    }

    kmer_colors = ['red','green','blue','purple','cyan','orange','magenta','black','hotpink']
    (num_proteins,C,ig) = hit_matrix.shape
    C = C-1
    V = min([9,len(kmers)])
    data = np.zeros((num_proteins,C,3),dtype='float')
    for i in range(V):
        data += hit_matrix[:,1:,i:i+1] * np.array(list(convert.to_rgb(kmer_colors[i]))) 
    
    idx = (hit_matrix[:,0,0]==1).nonzero()[0]
    data[idx,:,:] = data[idx,:,:] + (1-(data[idx,:,:].sum(2)>0)).reshape(idx.size,C,1) * background_colors['white']
    idx = (hit_matrix[:,0,0]==2).nonzero()[0]
    data[idx,:,:] = data[idx,:,:] + (1-(data[idx,:,:].sum(2)>0)).reshape(idx.size,C,1) * background_colors['offwhite']
#    idx = (hit_matrix[:,0,0]==3).nonzero()[0]
#    data[idx,:,:] = data[idx,:,:] + (1-(data[idx,:,:].sum(2)>0)).reshape(idx.size,C,1)*color_scheme['white']

    fig = plot.figure()
    im = fig.add_subplot(111)
    im.set_position([0.03,0.07,0.80,0.88])
    im.imshow(data,aspect='auto',interpolation='nearest')
    im.axis([0,hit_matrix.shape[1]-1,0,hit_matrix.shape[0]])
    im.set_xticks([0,hit_matrix.shape[1]-1])
    im.set_xticklabels((0,1))
    im.set_xlabel('Relative location')
    y_labels = ('Plant','Animal')
    y_label_loc = []
    for c in np.unique(hit_matrix[:,0,0]):
        y_label_loc.append(int(np.mean((hit_matrix[:,0,0]==c).nonzero()[0])))
    im.set_yticks(y_label_loc)
    im.set_yticklabels(y_labels, rotation=90)
    for line in im.get_yticklines():
        line.set_markersize(0)

    im.set_title('k = %d, m = %d' % (k,m))

    # a figtext bbox for legend
    kmer_locs = np.linspace(0.5+V/2*0.04,0.5-V/2*0.04,V)
    for kidx in range(V):
        kmer = kmers[kidx]
        try:
            plot.figtext(0.84, kmer_locs[kidx], kmer, fontsize=9, color=kmer_colors[kidx], horizontalalignment='left', verticalalignment='center')
        except IndexError:
            pdb.set_trace()

    fname = project_path + 'fig/' + virus_family + '_kmer_visualization_%d_%d_%d.pdf' % (k,m,fold)
    fig.savefig(fname,dpi=(300),format='pdf')


if __name__=="__main__":
#    project_path = '/proj/ar2384/picorna/'
    project_path = '../'
    data_path = project_path + 'cache/'
    virus_family = 'rhabdo'

    # k,m values
    (k, m, fold) = map(int,sys.argv[1:4])
    
    # load classes
    classes = dict()
    c = open(project_path + 'data/' + virus_family + '_classes.csv','r')
    for line in c:
        row = line.strip().split(',')
        virus_name = ' '.join(row[0].split()[1:])
        classes[row[0].split()[0]] = [virus_name,int(row[1])]

    # load kmers
    kmers = []
    f = open(data_path + virus_family + '_decisiontree_%d_%d_%d.pkl' % (k,m,fold),'r')
    decision_tree = cPickle.load(f)
    order = cPickle.load(f)
    f.close()
    [kmers.append(decision_tree[o][0][0]) for o in order if decision_tree[o][0][0] not in kmers] 

    # load protein strings
    proteins = []
    protein = 'A'
    label = 0
    viruses = []
    p = open(project_path + 'data/' + virus_family + 'virus-proteins.fasta','r')
    # THIS NEEDS TO BE GENERALIZED!
    if virus_family == 'picorna':
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
    elif virus_family == 'rhabdo':
        for line in p:
            if 'NC_' in line or '>' in line:
                proteins.append([protein,label])
                row = line.strip().split(',')
                virus_name = ' '.join(row[0].split()[1:])
                virus_id = row[0].split()[0][1:]
                viruses.append(virus_id)
                label = classes[virus_id][1]
                protein = ''
            else:
                protein = line.strip()
    c.close()
    p.close()
    proteins.pop(0)
    hit_matrix = compile_hit_matrix(proteins,kmers,m)

    # save compiled data
    f = open(data_path + virus_family + '_hitmatrix_%d_%d_%d.pkl' % (k,m,fold),'w')
    cPickle.Pickler(f,protocol=2).dump(hit_matrix)
    cPickle.Pickler(f,protocol=2).dump(viruses)
    cPickle.Pickler(f,protocol=2).dump(classes)
    f.close()
    """

    # load data
    f = open(data_path + virus_family + '_hitmatrix_%d_%d_%d.pkl' % (k,m,fold),'r')
    hit_matrix = cPickle.load(f)
    viruses = cPickle.load(f)
    classes = cPickle.load(f)
    f.close()
    """
    
    sort_indices = hit_matrix[:,0,0].argsort()
    sort_virus_id = [viruses[i] for i in sort_indices]
    sort_viruses = [classes[v][0] for v in sort_virus_id]
    plot_hit_matrix(hit_matrix[sort_indices,:,:],k,m,fold,kmers,virus_family,project_path)
