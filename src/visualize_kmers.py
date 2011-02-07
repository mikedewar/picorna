import numpy as np
import cPickle
import matplotlib.pyplot as plot
from matplotlib.transforms import Bbox
from matplotlib.colors import colorConverter as convert
import pdb
import sys

# compile hit matrix
def compile_hit_matrix(sequencess, kmers, m):
    col_size = 500
    N_sequences = len(sequences)
    N_kmers = len(kmers)
    hit_matrix = np.zeros((N_sequences,col_size+1,N_kmers),dtype='float')
    kmer_length = len(kmers[0])

    # this dictionary stores a set of alpha values
    # (transparency) for each mismatch.
    options = dict()
    for i in range(m+1):
        options[i] = 1.

    for index, seq in enumerate(sequences):
        # first column stores the virus class
        hit_matrix[index,0,:] = seq[1]
        sequence = seq[0]
        sequence_length = len(sequence)
        for c in range(sequence_length-kmer_length+1):
            for kidx, kmer in enumerate(kmers):
                mismatch = (np.array(list(sequence[c:c+kmer_length]))!=np.array(list(kmer))).sum()
                try:
                    value = options[mismatch]
                except KeyError:
                    continue
                left_col = int(c * float(col_size) / (sequence_length-kmer_length+1)) + 1
#               right_col = max([int((c+kmer_length) * float(col_size) / (sequence_length-kmer_length+1)),left_col+1])
                right_col = min([left_col+2,sequence_length-kmer_length+1])
                hit_matrix[index,left_col:right_col,kidx] = value 

    return hit_matrix

# plot hit matrix
def plot_hit_matrix(hit_matrix, k, m, fold, kmers, virus_family, sequence_type, project_path):
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

    fname = project_path + 'fig/%s_%s_kmer_visualization_%d_%d_%d.pdf' % (virus_family, sequence_type, k, m, fold)
    fig.savefig(fname,dpi=(300),format='pdf')


if __name__=="__main__":
    project_path = '/proj/ar2384/picorna/'
    virus_family = 'rhabdo'
    sequence_type = 'protein'
    data_path = project_path + 'cache/%s_temp/' % virus_family

    # k,m values
    (k, m, T, fold) = map(int,sys.argv[1:5])
    
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
    [kmers.append(decision_tree[o][0][0]) for o in order[:T] if decision_tree[o][0][0] not in kmers] 

    # load virus sequences
    sequences = []
    sequence = 'A'
    label = 0
    viruses = []
    p = open(project_path + 'data/' + virus_family + 'virus-proteins.fasta','r')
    
    if sequence_type == 'protein':
        for line in p:
            if 'NC_' in line or 'virus' in line:
                sequences.append([sequence,label])
                row = line.strip().split(',')
                virus_name = ' '.join(row[0].split()[1:])
                virus_id = row[0].split()[0]
                viruses.append(virus_id)
                label = classes[virus_id][1]
                sequence = ''
            elif '>' in line:
                continue
            else:
                sequence += line.strip()
    elif sequence_type == 'dna':
        for line in p:
            if 'virus' in line:
                sequences.append([sequence,label])
                row = line.strip().split(',')
                virus_name = ' '.join(row[0].split()[1:])
                virus_id = row[0].split()[0][1:]
                viruses.append(virus_id)
                label = classes[virus_id][1]
                sequence = ''
            else:
                sequence += line.strip()
    c.close()
    p.close()
    sequences.pop(0)
    hit_matrix = compile_hit_matrix(sequences,kmers,m)

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
    plot_hit_matrix(hit_matrix[sort_indices,:,:], k, m, fold, kmers, virus_family, sequence_type, project_path)
