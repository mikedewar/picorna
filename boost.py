import numpy as np
import scipy.weave as weave
#import weave
import cPickle
import time
import pdb

# machine epsilon
EPS = np.finfo(np.double).tiny

def adaboostMH(X,Y,x,y,params,kmer_dict,model='stump'):
    """
    Input:
        X : DxN array (Train data) 
        Y : KxN array (Train labels)
        x : Dxn array (Test data)
        y : Kxn array (Test labels)
        params : tuple (fold index, kmer length, mismatch)
        kmer_dict : a dictionary mapping row id
                    to kmers.
        model : string
            can be "tree" or "stump"
    """

    X = X.astype('float')
    Y = Y.astype('float')
    (D,N) = X.shape
    K = Y.shape[0]
    n = x.shape[1]

    f = params[0]
    k = params[1]
    m = params[2]
    
    # Number of boosting rounds
    T = 25

    """
    creating output files
    onfname - test/train errors and the selected feature 
             at each round is output in this file
    tnfname - the decision tree after T rounds of boosting
            is output in this file
    dfname - a general dump of the test/train predictions 
        for all T rounds is output in this file
    """    
    filedir = '/proj/ar2384/picorna/Adaboost/'
    filetag = model+'_%d_%d_%d' % (k,m,f)
    onfname = filedir+'error'+filetag+'.txt'
    tnfname = filedir+'decision'+filetag+'.pkl'
    dfname = filedir+'dump'+filetag+'.pkl'
    
    # Initializing weight over examples - Uniform distribution
    w = np.ones(Y.shape, dtype='float')/(N*K)
    
    #Data structures to store output from boosting at each round. 
    #dectree - a list of all nodes (and their attributes) in the decision tree
    #Tpred/tpred - stores the output of the decision tree at each round (train/test samples)
    Phidict = dict(); phidict = dict()
    dectree = dict(); order = []
    Tpred = np.zeros((K,N,T+1), dtype='float32')
    tpred = np.zeros((K,n,T+1), dtype='float32')
    rocacc = np.zeros((T+1,5),dtype='float32')

    starttime = time.time()
    # root decision function/prediction node.
    # root decision function always outputs 1.
    v = ((w*Y).sum(1)>0)*2.-1.
    v = v.reshape(K,1)
    # gamma = 'edge' of the weak rule
    gamma = (w*Y*v).sum()
    # a = coefficient of weak rule
    a = 0.5*np.log((1+gamma)/(1-gamma))

    # update decision tree and prediction list.
    Phi = np.ones((1,N), dtype='float32')
    phi = np.ones((1,n), dtype='float32')
    Hweakrule = v*Phi
    hweakrule = v*phi
    # Phidict keys = feature ids
    # Phidict values = [\phi(x), feature wt, >/< decision, weak rule's output]
    Phidict[-1] = [[Phi,a,Hweakrule]]
    phidict[-1] = [[phi,a,hweakrule]]
    dectree[-1] = [-1,[a,[],v]]

    # compute the prediction output by the decision
    # tree for all train/test samples
    train_pred = np.zeros((K,N), dtype='float32')
    test_pred = np.zeros((K,n), dtype='float32')
    for kidx in Phidict.keys():
        for aidx in range(len(Phidict[kidx])):
            train_pred = train_pred + Phidict[kidx][aidx][1]*Phidict[kidx][aidx][2]
            test_pred = test_pred + phidict[kidx][aidx][1]*phidict[kidx][aidx][2]

    # store the real-valued prediction
    Tpred[:, :, 0] = train_pred
    tpred[:, :, 0] = test_pred

    # compute classification error at round 0
    rocacc[0,1], rocacc[0,3] = classification_error(train_pred,test_pred,Y,y,0.)
    duration = time.time() - starttime
    
    # write some output to file
    # file format: boosting round, k-mer selected, 
    # train roc, train error, test roc, test error, time elapsed
    owrite = open(onfname,'w')
    to_write = [-1, 'root', 'None', 0.5, rocacc[0,1], 0.5, rocacc[0,3], duration]
    owrite.write('\t'.join(map(str,to_write))+'\n')
    owrite.close()
    print to_write

    # update weights
    wnew = w*np.exp(-a*Hweakrule*Y)
    wnew = wnew/wnew.sum()
    w = wnew.copy()

    # starting boosting rounds
    for t in range(T):
        starttime = time.time()

        # choose the appropriate (leaf+weak rule) for the next prediction function
        #pstar, cstar, pastar, cvalue, Z = py_get_weak_rule(X, Y, Phidict, w, model)
        pstar, cstar, pastar, cvalue, Z = weave_get_weak_rule(X, Y, Phidict, w, model)
        PX = (X[cstar:cstar+1, :]<cvalue)*1
        px = (x[cstar:cstar+1, :]<cvalue)*1
        order.append(t)

        # Updating Tree and prediction dictionary
        Phidict[t] = []; phidict[t] = []
        # FIXME: need to replace with actual k-mer
        dectree[t] = [[kmer_dict[cstar],cvalue]]
        dectree[pstar][pastar+1][1].append(t)
        Hweakrule = np.zeros((K,N),dtype='float')
        hweakrule = np.zeros((K,n),dtype='float')
        ans = [0,1]

        for aidx in ans:
            # compute output of decision function
            Phi = Phidict[pstar][pastar][0]*(aidx+((-1.)**aidx)*PX)
            phi = phidict[pstar][pastar][0]*(aidx+((-1.)**aidx)*px)
            # calculate optimal value of alpha for that decision
            wYP = w*Y*Phi
            vstar = ((wYP.sum(1) > 0)*2.-1.).reshape(K,1)
            gamma = (wYP*vstar).sum()
            a = 0.5*np.log((1+gamma)/(1-gamma))

            # compute f(x) = \alpha * \phi(x) * v for each decision node
            Hweakrule += a*vstar*Phi

            # Update Tree and prediction dictionary
            Phidict[t].append([Phi,a,vstar*Phi])
            phidict[t].append([phi,a,vstar*phi])
            dectree[t].append([a,[],vstar])
        
        # Update example weights
        wnew = w*np.exp(-1.*Hweakrule*Y)
        wnew = wnew/wnew.sum()
        w = wnew.copy()

        # Calculate train and test predictions and errors
        train_pred = np.zeros((K,N), dtype='float32')
        test_pred = np.zeros((K,n), dtype='float32')
        for kidx in Phidict.keys():
            for aidx in range(len(Phidict[kidx])):
                train_pred = train_pred + Phidict[kidx][aidx][1]*Phidict[kidx][aidx][2]
                test_pred = test_pred + phidict[kidx][aidx][1]*phidict[kidx][aidx][2]

        Tpred[:, :, t+1] = train_pred
        tpred[:, :, t+1] = test_pred
        rocacc[t+1,0], rocacc[t+1,2], rocacc[t+1,4] = roc_auc(train_pred, test_pred, Y, y)
        rocacc[t+1,1], rocacc[t+1,3] = classification_error(train_pred, test_pred, Y, y, rocacc[t+1,4])
        duration = time.time() - starttime

        # output data
        owrite = open(onfname,'a')
        to_write = [t, kmer_dict[cstar], cvalue, rocacc[t+1,0], rocacc[t+1,1],
                rocacc[t+1,2], rocacc[t+1,3], duration]
        owrite.write('\t'.join(map(str,to_write))+'\n')
        owrite.close()
        print to_write
    
    # output decision tree
    twrite = open(tnfname,'w')
    cPickle.Pickler(twrite,protocol=2).dump(dectree)
    cPickle.Pickler(twrite,protocol=2).dump(order)
    twrite.close()

    # dump predictions for more analysis
    dwrite = open(dfname,'w')
    cPickle.Pickler(dwrite,protocol=2).dump(Tpred)
    cPickle.Pickler(dwrite,protocol=2).dump(tpred)
    cPickle.Pickler(dwrite,protocol=2).dump(rocacc)
    dwrite.close()

def roc_auc(train_pred,test_pred,Y,y,threshold='None'):
    """
    Computes the ROC curve and the area
    under that curve, as a measure of classification
    accuracy. If a threshold is specified, the
    (precision,recall) for the given threshold is returned.
    """
    
    if threshold=='None':
        values = np.sort(np.unique(train_pred.ravel()))
        indices = np.arange(1,values.size-2,2)
        Thresholds = list(values[indices])
        Thresholds.extend([values[1],values[-2]])
        Thresholds.sort(reverse=True)
        
        values = np.sort(np.unique(test_pred.ravel()))
        indices = np.arange(1,values.size-2,2)
        thresholds = list(values[indices])
        thresholds.extend([values[1],values[-2]])
        thresholds.sort(reverse=True)
    else:
        Thresholds = [threshold]
        thresholds = [threshold]

    TPR = np.zeros((len(Thresholds)+2,2),dtype='float')
    tPR = np.zeros((len(thresholds)+2,2),dtype='float')
    TPR[0,:] = np.array([0,0]); TPR[-1,:] = np.array([1,1])
    tPR[0,:] = np.array([0,0]); tPR[-1,:] = np.array([1,1])

    for tidx in range(len(Thresholds)):
        P = (train_pred>Thresholds[tidx])
        true_positive = (P*Y==1).sum()
        real_positive = 0.5*(1+Y).sum()

        # precision-recall
#        pred_positive = np.float(P.sum())
#        TPR[tidx,:] = np.array([true_positive/pred_positive,true_positive/real_positive])

        # roc
        false_positive = (P*Y==-1).sum()
        real_negative = 0.5*(1-Y).sum()
        TPR[tidx+1,:] = np.array([false_positive/real_negative, true_positive/real_positive])

    for tidx in range(len(thresholds)):
        p = (test_pred>thresholds[tidx])
        true_positive = (p*y==1).sum()
        real_positive = 0.5*(1+y).sum()

        # precision-recall
#        pred_positive = np.float(p.sum())
#        tPR[tidx,:] = np.array([true_positive/pred_positive,true_positive/real_positive])

        # roc
        false_positive = (p*y==-1).sum()
        real_negative = 0.5*(1-y).sum()
        tPR[tidx+1,:] = np.array([false_positive/real_negative, true_positive/real_positive])

    # compute area under the curve using trapezoidal methods
    arTPR = np.trapz(TPR[:,1],TPR[:,0])
    artPR = np.trapz(tPR[:,1],tPR[:,0])

    # a simple way to pick a threshold on the roc curve
    dist = np.abs(TPR[:,0]-TPR[:,1])

    return arTPR, artPR, Thresholds[dist.argmax()-1]


def classification_error(train_pred, test_pred, Y, y, thresh):
    """
    Input:
        train_pred : KxN array (real-valued predictions)
        test_pred : Kxn array (real-valued predictions)
        Y : KxN array ({1,-1})
        y : Kxn ({1,-1})
        thresh : int (cut-off for ordinal predictions)
    """

    (K,N) = Y.shape
    n = y.shape[1]
    K = float(K)

    # train accuracy
#    P = (train_pred>thresh)*1.
#    trainacc = np.mean(((P*Y>0).sum(0))/(P.sum(0)+EPS))
    trainacc = (np.abs(train_pred.argmax(0)-Y.argmax(0))>0).sum()/float(N)

    # test accuracy     
#    p = (test_pred>thresh)*1.
#    testacc = np.mean(((p*y>0).sum(0))/(p.sum(0)+EPS))
    testacc = (np.abs(test_pred.argmax(0)-y.argmax(0))>0).sum()/float(n)

    return trainacc, testacc


def weave_get_weak_rule(X, Y, Phidict, w, m):
    code =     """
        double Z = 0;
        int pstar, cstar, pastar, cvalue;
        int D = NX[0];
        int N = NX[1];
        int P = Nfi[0];
        int K = NY[0];
        int T = Nthresholds[0];
        double zg, zl, Wg, Wl;
        double zr, tr;

        for (int p=0;p<P;p++) {
            for (int d=0;d<D;d++) {
                // Loop over thresholds
                zr = 0;
                for (int t=0;t<T;t++) {
                    double threshold = thresholds[t];
                    zg = 0;
                    zl = 0;

                    for (int k=0;k<K;k++) {
                        Wg = 0;
                        Wl = 0;
                        for (int n=0;n<N;n++) {
                            if (X[d*N+n]>=threshold)
                                Wg += w[k*N+n] * Y[k*N+n] * fi[p*N+n];
                            else
                                Wl += w[k*N+n] * Y[k*N+n] * fi[p*N+n];
                        }
                        if (Wg>0)
                            zg += Wg;
                        else
                            zg -= Wg;
                        if (Wl>0)
                            zl += Wl;
                        else
                            zl -= Wl;
                    }

                    if (fmax(zg,zl)>zr) {
                        zr = fmax(zg,zl);
                        tr = threshold;
                    }
            
                    if (zr>Z) {
                        Z = zr;
                        cvalue = threshold;
                        cstar = d;
                        pstar = p;
                    }
                }
                Zres[p*D*2+d*2] = zr;
                Zres[p*D*2+d*2+1] = tr;
            }
        }

        results[0] = pstar;
        results[1] = cstar;
        results[2] = cvalue;
        """

    support = """
        #include <math.h>
        """

    # the python code that calls the C++ number-cruncher
    # via weave.inline
    (D,N) = X.shape
    K = Y.shape[0]
    fi = np.zeros((0,N),dtype='int')

    if m=='tree':
        pkeys = Phidict.keys()
        pkeys.sort()
        porder = []
        for key in pkeys:
            pdec = range(len(Phidict[key]))
            for pd in pdec:
                fi = np.vstack((fi,Phidict[key][pd][0]))
                porder.append(key)
        results = np.zeros((3,),dtype='int')
        Zres = np.zeros((len(porder),D,2),dtype='float')
        fi = fi.astype('float')
        thresholds = np.unique(X[:])

        weave.inline(code, ['X','Y','fi','w','thresholds','Zres','results'], support_code = support, verbose=2, compiler='gcc')
    elif m=='stump':
        pass

    pstar = porder[results[0]]
    cstar = results[1]
    if results[0]:
        pastar = 1-results[0]%2
    else:
        pastar = results[0]
    cvalue = results[2]

    return pstar, cstar, pastar, cvalue, Zres


def py_get_weak_rule(X, Y, Phidict, w, m):
    """
    Input:
        X : DxN array
        Y : KxN array
        Phidict : dict (output of weak-rules at each node of the tree)
        w : KxN array (weights over examples that sum to 1)
        m : string
            can be "tree" or "stump"
    """
    
    (D,N) = X.shape
    K = Y.shape[0]

    if m=='tree':
        pkeys = Phidict.keys()
        pkeys.sort()
        P = len(pkeys)
        Z = np.zeros((P,D,4),dtype='float32')
        for p in range(P):
            key = pkeys[p]
            plen = len(Phidict[key])
            for d in range(D):
                thresholds = np.unique(X[d:d+1,:])
                tlen = thresholds.size
                # dim0 = </> for parent; dim1 = </> for child; dim2 = threshold
                z = np.zeros((plen,2,tlen),dtype='float')
                for tidx in range(tlen):
                    threshold = thresholds[tidx]
                    for pd in range(plen):
                        # 'less-then' decision
                        fi = Phidict[key][pd][0]*(X[d:d+1,:]<threshold)
                        wYf = w*Y*fi
                        vstar = ((wYf.sum(1)>0)*2.-1.).reshape(K,1)
                        z[pd,0,tidx] = (wYf*vstar).sum()

                        # greater-than decision
                        fi = Phidict[key][pd][0]*(X[d:d+1,:]>=threshold)
                        wYf = w*Y*fi
                        vstar = ((wYf.sum(1)>0)*2.-1.).reshape(K,1)
                        z[pd,1,tidx] = (wYf*vstar).sum()

                Z[p,d,0] = z.max()
                (Z[p,d,1],cl,Z[p,d,2]) = np.argwhere(z==z.max())[0]
                Z[p,d,2] = thresholds[Z[p,d,2]]
                    
        pstar, cstar = np.argwhere(Z[:,:,0]==Z[:,:,0].max())[0]
        pastar = int(Z[pstar,cstar,1])
        cvalue = int(Z[pstar,cstar,2])
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

    return pstar, cstar, pastar, cvalue, Z[:,:,[0,2]]
