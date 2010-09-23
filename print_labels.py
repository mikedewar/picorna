import numpy as np
import cPickle

k = 7
m = 2

# load data
f = open('picorna_virii_data_%d_%d.pkl' % (k,m),'r')
X = cPickle.load(f)
Y = cPickle.load(f)
f.close()

# load predicted labels
f = open('Adaboost/picorna_virii_test_output_%d_%d.pkl' % (k,m),'r')
Fidx = cPickle.load(f)
predicted_labels = cPickle.load(f)
f.close()

# load virus classes
classes = dict()
c = open('classes.csv','r')
for line in c:
    row = line.strip().split(',')
    virus_name = ' '.join(row[0].split()[1:])
    classes[row[0].split()[0]] = [virus_name,int(row[1])]

label_dict = {
    1 : 'Invertebrate',
    2 : 'Plant',
    3 : 'Vertebrate'
}

# write a text file listing true & predicted 
# labels of viruses.
f = open('compare_virus_labels_%d_%d.txt' % (k,m),'w')
p = open('picornavirus-proteins.fasta','r')
idx = 0
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

p.close()
f.close()
