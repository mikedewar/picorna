import numpy as np
import cPickle
import pdb
import sys

(k,m,cut_off) = map(int,sys.argv[1:4])
virus_family = 'rhabdo'
project_path = '/proj/ar2384/picorna/'

# load data
f = open(project_path + 'cache/%s_virii_data_%d_%d.pkl' % (virus_family,k,m),'r')
X = cPickle.load(f)
Y = cPickle.load(f)
f.close()

# load predicted labels
f = open(project_path + 'cache/%s_virii_test_output_%d_%d.pkl' % (virus_family,k,m),'r')
Fidx = cPickle.load(f)
predicted_labels = cPickle.load(f)
f.close()

# load virus classes
classes = dict()
c = open(project_path + 'data/%s_classes.csv' % virus_family,'r')
for line in c:
    row = line.strip().split(',')
    virus_name = ' '.join(row[0].split()[1:])
    classes[row[0].split()[0]] = [virus_name,int(row[1])]

#label_dict = {
#    1 : 'Invertebrate',
#    2 : 'Plant',
#    3 : 'Vertebrate'
#}
label_dict = {
    1 : 'Plant',
    2 : 'Animal'
}

# write a text file listing true & predicted 
# labels of viruses.
f = open(project_path + 'fig/%s_compare_virus_labels_%d_%d.txt' % (virus_family,k,m),'w')
p = open(project_path + 'data/%svirus-proteins.fasta' % virus_family,'r')
idx = 0
if virus_family == 'picorna':
    for line in p:
        if 'NC_' in line:
            f.write('\n')
            row = line.strip().split(',')
            virus_id = row[0].split()[0]
            true_label = classes[virus_id][1]
            virus_name = ' '.join(row[0].split()[1:])
            f.write('True label : '+label_dict[true_label].upper()+'\t'+virus_name+'\n')
        elif '>' in line:
            row = line.strip()
            pred_label = predicted_labels[idx,-1]+1
            f.write('Prediction : '+label_dict[pred_label].upper()+' using protein sequence '+row+'\n')
            idx += 1
elif virus_family == 'rhabdo':
    for line in p:
        if '>' in line:
            f.write('\n')
            row = line.strip().split(',')
            virus_id = row[0].split()[0][1:]
            virus_name = ' '.join(row[0].split()[1:])
            true_label = classes[virus_id][1]
            f.write('True label : '+label_dict[true_label].upper()+'\t'+virus_name+'\n')
        else:
            try:
                pred_label = predicted_labels[idx,cut_off]+1
            except IndexError:
                pdb.set_trace()
            f.write('Prediction : '+label_dict[pred_label].upper()+'\n')
            idx += 1
    
p.close()
f.close()
