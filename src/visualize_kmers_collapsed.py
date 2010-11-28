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
    hit_matrix = np.zeros((N_proteins,col_size+1),dtype='float')
    kmer_length = len(kmers[0][0])

    for pidx, protein in enumerate(proteins):
        # first column stores the virus class
        hit_matrix[pidx,0] = protein[1]
        protein = protein[0]
        protein_length = len(protein)
        for c in range(protein_length-kmer_length+1):
            for fold in range(len(kmers)):
                for kmer in kmers[fold]:
                    mismatch = (np.array(list(protein[c:c+kmer_length]))!=np.array(list(kmer))).sum()
                    try:
                        col = int(c * float(col_size) / (protein_length-kmer_length+1)) + 1
                        hit_matrix[pidx,col] += (mismatch<=m)
                    except KeyError:
                        continue
    
    # normalize by number of cv-folds
    hit_matrix[:,1:] = hit_matrix[:,1:]/len(kmers)

    return hit_matrix

# plot hit matrix
def plot_hit_matrix(hit_matrix,k,m,kmers,virus_family,project_path):
    background_colors = {
        'white' :   np.array([255,255,255]).reshape(1,3)/255.,
        'black' :   np.array([0,0,0]).reshape(1,3)/255.,
         'grey' :   np.array([38,38,38]).reshape(1,3)/255.,
     'darkgrey' :   np.array([18,18,18]).reshape(1,3)/255.,
     'offwhite' :   np.array([235,235,235]).reshape(1,3)/255.,
    }

    (num_proteins,num_cols) = hit_matrix.shape
    num_cols = num_cols-1
    data = np.zeros((num_proteins,num_cols,3),dtype='float')
    data[:,:,0] = hit_matrix[:,1:] 
    data = data * np.array(list(convert.to_rgb('red'))).reshape(1,1,3)
    pdb.set_trace()
    
    idx = (hit_matrix[:,0]==1).nonzero()[0]
    data[idx,:,:] = data[idx,:,:] + (hit_matrix[idx,1:]==0).reshape(idx.size,num_cols,1) * background_colors['black'].reshape(1,1,3)
    idx = (hit_matrix[:,0]==2).nonzero()[0]
    data[idx,:,:] = data[idx,:,:] + (hit_matrix[idx,1:]==0).reshape(idx.size,num_cols,1) * background_colors['grey'].reshape(1,1,3)
#    idx = (hit_matrix[:,0,0]==3).nonzero()[0]
#    data[idx,:,:] = data[idx,:,:] + (1-(data[idx,:,:].sum(2)>0)).reshape(idx.size,C,1)*color_scheme['white']

    fig = plot.figure()
    im = fig.add_subplot(111)
    im.set_position([0.03,0.07,0.97,0.88])
    im.imshow(data,aspect='auto',interpolation='nearest')
    im.axis([0,hit_matrix.shape[1]-1,0,hit_matrix.shape[0]])
    im.set_xticks([0,hit_matrix.shape[1]-1])
    im.set_xticklabels((0,1))
    im.set_xlabel('Relative location')
    y_labels = ('Plant','Animal')
    y_label_loc = []
    for c in np.unique(hit_matrix[:,0]):
        y_label_loc.append(int(np.mean((hit_matrix[:,0]==c).nonzero()[0])))
    im.set_yticks(y_label_loc)
    im.set_yticklabels(y_labels, rotation=90)
    for line in im.get_yticklines():
        line.set_markersize(0)

    im.set_title('k = %d, m = %d' % (k,m))

    fname = project_path + 'fig/' + virus_family + '_kmer_visualization_collapsed_%d_%d.pdf' % (k,m)
    fig.savefig(fname,dpi=(300),format='pdf')


if __name__=="__main__":
#    project_path = '/proj/ar2384/picorna/'
    project_path = '../'
    data_path = project_path + 'cache/'
    virus_family = 'rhabdo'

    # k,m values
    (k, m, cut_off) = map(int,sys.argv[1:4])
    
    # load classes
    classes = dict()
    c = open(project_path + 'data/' + virus_family + '_classes.csv','r')
    for line in c:
        row = line.strip().split(',')
        virus_name = ' '.join(row[0].split()[1:])
        classes[row[0].split()[0]] = [virus_name,int(row[1])]

    # load kmers
    folds = 5
    kmers = []
    for fold in range(folds):
        f = open(data_path + virus_family + '_decisiontree_%d_%d_%d.pkl' % (k,m,fold),'r')
        decision_tree = cPickle.load(f)
        order = cPickle.load(f)
        f.close()
        [kmers.append(list(set([decision_tree[o][0][0] for o in order[:cut_off]])))]

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
    f = open(data_path + virus_family + '_hitmatrix_collapsed_%d_%d.pkl' % (k,m),'w')
    cPickle.Pickler(f,protocol=2).dump(hit_matrix)
    cPickle.Pickler(f,protocol=2).dump(viruses)
    cPickle.Pickler(f,protocol=2).dump(classes)
    f.close()
    """

    # load data
    f = open(data_path + virus_family + '_hitmatrix_collapsed_%d_%d.pkl' % (k,m),'r')
    hit_matrix = cPickle.load(f)
    viruses = cPickle.load(f)
    classes = cPickle.load(f)
    f.close()
    """
    
    sort_indices = hit_matrix[:,0].argsort()
    sort_virus_id = [viruses[i] for i in sort_indices]
    sort_viruses = [classes[v][0] for v in sort_virus_id]
    plot_hit_matrix(hit_matrix[sort_indices,:],k,m,kmers,virus_family,project_path)
