import numpy as np
import splitdata
import boost
import pdb

data = np.load('/proj/ar2384/picorna/picorna_virii_data.npz')
#data = np.load('MHtestdata.npz')
Xt = data['X']
Yt = data['Y']

# make Xt, Yt memory-efficient
# would be good to do this from the get go
Xt = Xt.astype('float32')
Yt = Yt.astype('float32')

# number of folds of cross validation
Nfold = 10

# split the data indices into 10 random disjoint sets
Fidx = splitdata.cv_multiclass_fold(Yt,Nfold)

for f in range(3):
    # using each set as the test set and the rest as train sets
    # split the data and run boosting
    X, Y, x, y, Idx = splitdata.cv_split(Xt,Yt,Fidx[f])
    boost.adaboostMH(X, Y, x, y, f, model='tree')
