import numpy as np
import cPickle
import time
import pdb

def adaboostMH(X,Y,x,y,f,model='stump'):
    """
    X : DxN array 
        binary data matrix
    Y : KxN array
        binary matrix of labels
    x : ? x n array
        
    y : ? x n array
        
    f : 
        
    model : string
        can be "tree" or "stump"
    """
	(D,N) = X.shape
	K = Y.shape[0]
	n = x.shape[1]
	
	# Number of boosting rounds
	T = 100

	# creating output files
	filedir = './Adaboost/'
	filetag = model+'_'+str(f)
	onfname = filedir+'error'+filetag+'.txt'
	tnfname = filedir+'decision'+filetag+'.txt'
	dfname = filedir+'dump'+filetag+'.dump'
	
	# Initial weight over examples - Uniform
	w = np.ones(Y.shape, dtype='float32')/(N*K)
    # w = (1+Y)/(4*N) + (1-Y)/(4*N*(K-1))

	# Initialize decision tree, prediction list objects
	Tpredlist = []
	tpredlist = []
	dectree = []
	Tpred = np.zeros((N,T+1), dtype='float32')
	tpred = np.zeros((n,T+1), dtype='float32')

	starttime = time.time()
	# root decision function/prediction node.
	# root decision function always gives true.
	Wmatch = (w*(Y>0)).sum(1)
	Wmismatch = (w*(Y<0)).sum(1)
	v = (Wmatch-Wmismatch>0)*2.-1.
	v = v.reshape(K,1)
	# gamma = 'edge' of the weak rule
	gamma = (w*Y*v).sum()
	# a = coefficient of weak rule
	a = 0.5*np.log((1+gamma)/(1-gamma))

	# update decision tree and prediction list.
	Philist = dict()
	Philist[-1] = np.ones((1, N), dtype='float32')
	philist = dict()
	philist[-1] = np.ones((1, n), dtype='float32')
	Hweakrule = v*Philist[-1]
	hweakrule = v*philist[-1]
	Tpredlist = [[1, a, Hweakrule, 0]]
	tpredlist = [[1, a, hweakrule, 0]]
	dectree = [[1, [a, []], -1]]
	dlen = 1; plen = 1; clen = D

	# training and test error
	train_pred = np.zeros((K, N), dtype='float32')
	test_pred = np.zeros((K, n), dtype='float32')
	for idx in range(plen):
		train_pred = train_pred + Tpredlist[idx][1]*Tpredlist[idx][2]
		test_pred = test_pred + tpredlist[idx][1]*tpredlist[idx][2]

	Tpred[:, 0] = np.argmax(train_pred, 0)
	tpred[:, 0] = np.argmax(test_pred, 0)
	trainerr, testerr = compute_error(train_pred, test_pred, Y, y)
	duration = time.time() - starttime
    
	# write output to file
	owrite = open(onfname,'a')
	to_write = [
	    -1, 
	    a, 
	    (train_pred*Y).sum()/float(K*N), 
	    trainerr, 
	    (test_pred*y).sum()/float(K*n), 
	    testerr, 
	    duration
	]
	owrite.write('\t'.join([str(s) for s in to_write]))
	owrite.close()
	print to_write
	# update weights
	Wt = []
	wnew = w*np.exp(-a*Hweakrule*Y)
	wnew = wnew/wnew.sum()
	Wt.append(wnew)
	w = wnew

	# starting boosting rounds
	for t in range(T):
		starttime = time.time()

		# choose the appropriate (leaf+weak rule) for the next prediction 
		# function
		pstar, cstar, astar = get_weak_rule(X, Y, Philist, w, model)
		if astar:	# negations
			Philist[t] = Philist[pstar]*(1-X[cstar:cstar+1, :])
			philist[t] = philist[pstar]*(1-x[cstar:cstar+1, :])
		else:
			Philist[t] = Philist[pstar]*X[cstar:cstar+1, :]
			philist[t] = philist[pstar]*x[cstar:cstar+1, :]	
		Phi = Philist[t]*2.-1.; phi = philist[t]*2.-1.

		# calculate optimal value of alpha (scalar)
		Wmatch = (w*(Phi*Y>0)).sum(1)
		Wmismatch = (w*(Phi*Y<0)).sum(1)
		vstar = (Wmatch-Wmismatch>0)*2.-1.
		vstar = vstar.reshape(K,1)
		gamma = (w*Y*vstar*Phi).sum()
		a = 0.5*np.log((1+gamma)/(1-gamma))

		# output of new prediction rule
		Hweakrule = vstar*Phi
		hweakrule = vstar*phi

		# Updating Tree and prediction list
		if astar:	# negations
			Tpredlist.append([astar,a,Hweakrule,dlen])
			tpredlist.append([astar,a,hweakrule,dlen])
			dectree.append([astar,[a,[]],cstar])
		else:
			Tpredlist.append([0,a,Hweakrule,dlen])
			tpredlist.append([0,a,hweakrule,dlen])
			dectree.append([0,[a,[]],cstar])	
		dec = Tpredlist[pstar][3]
		dectree[dec][1][1].append(dlen)
		plen += 1; dlen += 1

		# Calculate train and test predictions and errors
		train_pred = np.zeros((K, N), dtype='float32')
		test_pred = np.zeros((K, n), dtype='float32')
		for idx in range(plen):
			train_pred = train_pred + Tpredlist[idx][1]*Tpredlist[idx][2]
			test_pred = test_pred + tpredlist[idx][1]*tpredlist[idx][2]

		Tpred[:, t+1] = np.argmax(train_pred, 0)
		tpred[:, t+1] = np.argmax(test_pred, 0)
		trainerr, testerr = compute_error(train_pred, test_pred, Y, y)

		# Update example weights
		wnew = w*np.exp(-a*Hweakrule*Y)
		wnew = wnew/wnew.sum()
		Wt.append(wnew)
		w = wnew
		# keep track of time
		duration = time.time() - starttime

		# output data
		owrite = open(onfname,'a')
		to_write = [
		    t,
		    a,
		    astar, mostansqs[cstar],
		    (train_pred*Y).sum()/float(K*N),
		    trainerr, 
		    (test_pred*y).sum()/float(K*n), 
		    testerr,
		    duration
		]
		owrite.write('\t'.join([str(s) for s in to_write]))
		owrite.close()
		print to_write
	
	# output decision tree
	twrite = open(tnfname,'a')
	for l in range(len(dectree)):
		twrite.write(str(dectree[l])+'\n')
	twrite.close()

	# dump predictions for more analysis
	dwrite = open(dfname,'a')
	cPickle.Pickler(dwrite,protocol=2).dump(Tpred)
	cPickle.Pickler(dwrite,protocol=2).dump(tpred)
	dwrite.close()

def compute_error(train_pred,test_pred,Y,y):
    """
    train_pred : 
    
    test_pred : 
    
    Y : KxN array
    
    y : ? x n
    """
	(K,N) = Y.shape
	n = y.shape[1]
	K = float(K)

	# naive sign-mismatch penalty
	# too stringent
	trainerr = (train_pred*Y<0).sum()/float(K*N)
	testerr = (test_pred*y<0).sum()/float(K*n)

	# rewards not-in-class prediction less than
	# in-class prediction
#	T = (train_pred/np.max(train_pred,0).reshape(1,N)==1)*(K/(K-1))
#	t = (test_pred/np.max(test_pred,0).reshape(1,n)==1)*(K/(K-1))
#	T = T-1./(K-1); t = t-1./(K-1)
#	trainerr = 1 - ((T*Y>0)*np.abs(T)).sum()/float(2*N)
#	testerr = 1 - ((t*y>0)*np.abs(t)).sum()/float(2*n)

	# rewards only in-class prediction
#	T = (train_pred/np.max(train_pred,0).reshape(1,N)==1)
#	t = (test_pred/np.max(test_pred,0).reshape(1,n)==1)
#	T = (train_pred>0)*1; t = (test_pred>0)*1
#	trainerr = (T*(0.5*(Y+1)*K-1.)/(K-1)).sum()/float(N)
#	testerr = (t*(0.5*(y+1)*K-1.)/(K-1)).sum()/float(n)

	return trainerr, testerr

def get_weak_rule(X,Y,Phidict,w,m):
    """
    X : DxN array
    
    Y : Kx? array
    
    Phidict : dict
    
    w : 
    
    m : string
        can be "tree" or "stump"
    """
    
	# This is going to be extremely slow for large
	# number of features. Need to rewrite in 
	# weave for major speed-up.
	# - Anil, 07/03

	(D,N) = X.shape
	K = Y.shape[0]

	if m=='tree':
		pkeys = Phidict.keys()
		pkeys.sort()
		P = len(pkeys)
		Z = np.zeros((P,D,2),dtype='float32')
		for p in range(P):
			key = pkeys[p]
			for d in range(D):
				fi = (Phidict[key]*X[d:d+1,:])*2.-1.
				Wp = (w*(fi*Y>0)).sum(1)
				Wm = (w*(fi*Y<0)).sum(1)
				vstar = (Wp-Wm>0)*2.-1.
				vstar = vstar.reshape(K,1)
				z0 = (w*Y*vstar*fi).sum()

				# negation
				fi = (Phidict[key]*(1-X[d:d+1,:]))*2.-1.
				Wp = (w*(fi*Y>0)).sum(1)
				Wm = (w*(fi*Y<0)).sum(1)
				vstar = (Wp-Wm>0)*2.-1.
				vstar = vstar.reshape(K,1)
				z1 = (w*Y*vstar*fi).sum()
				
				if z0>z1:
					Z[p,d,0] = z0; Z[p,d,1] = 0
				else:
					Z[p,d,0] = z1; Z[p,d,1] = 1

		pstar, cstar = np.argwhere(Z[:,:,0]==Z[:,:,0].max())[0]
		astar = int(Z[pstar,cstar,1])
		pstar = pkeys[pstar]

	elif m=='stump':
		Z = np.zeros((D,2),dtype='float32')
		for d in range(D):
			fi = (Phidict[-1]*X[d:d+1,:])*2.-1.
			Wp = (w*(fi*Y>0)).sum(1)
			Wm = (w*(fi*Y<0)).sum(1)
			vstar = (Wp-Wm>0)*2.-1.
			vstar = vstar.reshape(K,1)
			z0 = (w*Y*vstar*fi).sum()

			# negation
			fi = (Phidict[-1]*(1-X[d:d+1,:]))*2.-1.
                        Wp = (w*(fi*Y>0)).sum(1)
                        Wm = (w*(fi*Y<0)).sum(1)
                        vstar = (Wp-Wm>0)*2.-1.
                        vstar = vstar.reshape(K,1)
                        z1 = (w*Y*vstar*fi).sum()

			if z0>z1:
				Z[d,0] = z0; Z[d,1] = 0
			else:
				Z[d,0] = z1; Z[d,1] = 1
		
		pstar = -1
		cstar = np.argmax(Z[:,0])
		astar = int(Z[cstar,1])

	return pstar, cstar, astar
