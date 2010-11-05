import numpy as np
import random as rand
import pdb

def cv_multiclass_fold(Y,num_fold):
	"""
	split the data indices into test sets with elements
	of each class distributed among test sets
	in a balanced way.

	Arguments	
    =========    
    Y : KxN array	{1,-1}
	    indicates class membership
	num_fold : integer > 0
		number of cross-validation folds
		to split the data into.

	Returns
    =======	
    Sidx : list of length 'num_fold' 
	    list of num_fold lists of test data indices
	"""
	
	(K,N) = Y.shape
	indices = dict(); Nk = dict()
	for k in range(K):
		# select indices belonging to class k
		indices[k] = list((Y[k,:]==1).nonzero()[0])
		rand.shuffle(indices[k])
		Nk[k] = len(indices[k])/num_fold
	
	Sidx = []

	for k in range(K):
		for i in range(num_fold-1):
			# split class-k indices into num_fold random sets
			try:
				Sidx[i].extend(indices[k][Nk[k]*i:Nk[k]*(i+1)])
			except IndexError:
				Sidx.append([])
				Sidx[i].extend(indices[k][Nk[k]*i:Nk[k]*(i+1)])
		try:
			Sidx[num_fold-1].extend(indices[k][Nk[k]*(num_fold-1):])
		except IndexError:
			Sidx.append([])
			Sidx[num_fold-1].extend(indices[k][Nk[k]*(num_fold-1):])

	return Sidx

def cv_split(Xt,Yt,sdx):
	"""
	Given the test set indices,
	return train+test data

	Input:	Xt - DxNt array
			Nt is the total number of data points
			D is the dimensionality of the feature space
		Yt - KxNt array	{-1,1}
			indicates class membership
			K is the number of classes
		sdx - list
			indices selected to be the test set

	Output:	X - DxN array 
			(Train data)
		Y - KxN array 
			(Train labels)
		x - Dxn array 
			(Test data)
		y - Kxn array 
			(Test labels)
		Sdx - list
			indices selected to be the train set
	"""

	if Yt.shape[1]==1:
		tndx = np.zeros(Yt.shape,dtype='int')
	else:
		tndx = np.zeros((Yt.shape[1],1),dtype='int')
	tndx[sdx,0] = 1
	Tndx = 1 - tndx

	# training data
	X = Xt[:,Tndx.nonzero()[0]].astype('float')
	if Yt.shape[1]==1:
		Y = Yt[Tndx.nonzero()[0]]
	else:
		Y = Yt.T[Tndx.nonzero()[0],:].T

	# testing data
	x = Xt[:,tndx.nonzero()[0]].astype('float')
	if Yt.shape[1]==1:
		y = Yt[tndx.nonzero()[0]]
	else:
		y = Yt.T[tndx.nonzero()[0],:].T
	Sdx = Tndx.nonzero()[0]

	return X, Y, x, y, Sdx
