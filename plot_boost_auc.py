import numpy as np
import matplotlib.pyplot as plot
import pdb

ks = range(7,10)
#ks = [10]
ms = range(3)
fs = range(3)
T = 31

train_roc = np.zeros((len(ks),len(ms),len(fs),T),dtype='float')
train_roc[:,:,:,0] = 0.5
test_roc = np.zeros((len(ks),len(ms),len(fs),T),dtype='float')
test_roc[:,:,:,0] = 0.5

for k in ks:
    kidx = ks.index(k)
    for m in ms:
        midx = ms.index(m)
        for fidx in fs:
            filename = '/proj/ar2384/picorna/Adaboost/errortree_%d_%d_%d.txt' % (k,m,fidx)
            f = open(filename,'r')
            f.readline()
            t = 1
            for line in f:
                row = line.strip().split('\t')
                train_roc[kidx,midx,fidx,t] = row[2]
                test_roc[kidx,midx,fidx,t] = row[4]
                t += 1
            f.close()

train_roc = np.mean(train_roc,2)
test_roc = np.mean(test_roc,2)

X_max = 20
colors = ['r','b','g','k','m','c']
fig = []
im = []
for kidx in range(len(ks)):
    fig.append(plot.figure())
    im.append(fig[kidx].add_subplot(111))
    for midx in range(3):
        im[kidx].plot(range(X_max),test_roc[kidx,midx,:X_max],colors[midx]+'-o',label='m='+str(midx))

    im[kidx].axis([0,X_max,0.5,1])
    im[kidx].set_xlabel('Boosting Round',fontsize=12)
    im[kidx].set_ylabel('Mean Test AUC',fontsize = 12)
    im[kidx].set_title('k = '+str(ks[kidx]),fontsize=14)
    leg = im[kidx].legend(loc=4,labelspacing=0.1)
    for l in leg.get_texts():
        l.set_fontsize('10')

    outfile = '/proj/ar2384/picorna/Adaboost/boosterror_%d.pdf' % ks[kidx]
    fig[kidx].savefig(outfile,dpi=(100),format='pdf')
