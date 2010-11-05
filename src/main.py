import numpy as np
import cPickle
import splitdata
import boost
import pdb

k = 7
M = 4
for m in range(M):
    filename = '/proj/ar2384/picorna/picorna_virii_data_%d_%d.pkl' % (k,m)
    f = open(filename,'r')
    Xt = cPickle.load(f)
    Yt = cPickle.load(f)
    # quick hack to correct mis-labeling 
    # remove later
    Yt[:,[22,23]] = np.array([[-1.,-1.],[-1.,-1.],[1.,1.]])
    kmer_dict = cPickle.load(f)
    f.close()

    # make Xt, Yt memory-efficient
    Xt = Xt.astype('int16')
    Yt = Yt.astype('int16')
    Nt = Yt.shape[1]
    T = 15
    predicted_labels = np.zeros((Nt,T),dtype='int16')

    # number of folds of cross validation
    Nfold = 10

    # split the data indices into 10 random disjoint sets
    Fidx = splitdata.cv_multiclass_fold(Yt,Nfold)

    for fold in range(Nfold):
        params = (fold,k,m,T)
        # using each set as the test set and the rest as train sets
        # split the data and run boosting
        X, Y, x, y, Idx = splitdata.cv_split(Xt,Yt,Fidx[fold])
        predicted_labels = boost.adaboostMH(X, Y, x, y, predicted_labels, Fidx[fold], params, kmer_dict, model='tree')

    filename = '/proj/ar2384/picorna/picorna_virii_test_output_%d_%d.pkl' % (k,m)
    f = open(filename,'w')
    cPickle.Pickler(f,protocol=2).dump(Fidx)
    cPickle.Pickler(f,protocol=2).dump(predicted_labels)
    f.close()