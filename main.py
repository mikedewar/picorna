import numpy as np
import cPickle
import splitdata
import boost
import pdb

k = 10
m = 8
filename = '/proj/ar2384/picorna/picorna_virii_data_%d_%d.pkl' % (k,m)
f = open(filename,'r')
Xt = cPickle.load(f)
Yt = cPickle.load(f)
kmer_dict = cPickle.load(f)
f.close()

# make Xt, Yt memory-efficient
# would be good to do this from the get go
Xt = Xt.astype('int16')
Yt = Yt.astype('int16')

# number of folds of cross validation
Nfold = 10

# split the data indices into 10 random disjoint sets
Fidx = splitdata.cv_multiclass_fold(Yt,Nfold)

for f in range(3):
    params = (f,k,m)
    # using each set as the test set and the rest as train sets
    # split the data and run boosting
    X, Y, x, y, Idx = splitdata.cv_split(Xt,Yt,Fidx[f])
    boost.adaboostMH(X, Y, x, y, params, kmer_dict, model='tree')
