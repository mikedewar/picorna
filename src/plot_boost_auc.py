import numpy as np
import matplotlib.pyplot as plot
import pdb

# project path
project_path = '/proj/ar2384/picorna/'
virus_family = 'rhabdo'

ks = [25]
ms = range(5)
fs = range(5)
T = 41

train_roc = np.zeros((len(ks),len(ms),len(fs),T),dtype='float')
train_roc[:,:,:,0] = 0.5
test_roc = np.zeros((len(ks),len(ms),len(fs),T),dtype='float')
test_roc[:,:,:,0] = 0.5

for k in ks:
    kidx = ks.index(k)
    for m in ms:
        midx = ms.index(m)
        for fidx in fs:
            filename = project_path + 'cache/'+virus_family+'_errortree_%d_%d_%d.txt' % (k,m,fidx)
            f = open(filename,'r')
            f.readline()
            t = 1
            for line in f:
                row = line.strip().split('\t')
                train_roc[kidx,midx,fidx,t] = row[-5]
                test_roc[kidx,midx,fidx,t] = row[-3]
                t += 1
            f.close()

train_roc_mean = np.mean(train_roc,2)
train_roc_std = np.std(train_roc,2)
test_roc_mean = np.mean(test_roc,2)
test_roc_std = np.std(test_roc,2)

X_max = 15
colors = ['r','b','g','k','m','c']
fig = []
im = []
m_val = ms
bg_color = 'w' 
text_color = 'k'
for kidx in range(len(ks)):
    DPI = 300
    fig_resolution = (1024,768)
    fig_size = tuple([res/float(DPI) for res in fig_resolution])
    fig.append(plot.figure(figsize = fig_size, facecolor = bg_color, edgecolor = bg_color))
    im.append(fig[kidx].add_subplot(111, axisbg=bg_color))
    for m in m_val:
        midx = ms.index(m)
        im[kidx].errorbar(range(X_max), test_roc_mean[kidx,midx,:X_max], yerr=test_roc_std[kidx,midx,:X_max], marker='o', markersize=2, linestyle='-', linewidth=1, label='m = '+str(m))

    xtick_locs = range(0,X_max,2)
    xtick_labels = tuple(map(str,xtick_locs))
    im[kidx].set_xticks(xtick_locs)
    im[kidx].set_xticklabels(xtick_labels, color=text_color, fontsize=6, verticalalignment='center')
    im[kidx].set_xlabel('Boosting Round', fontsize=6, color=text_color)

    ytick_locs = [x*0.1 for x in range(5,11)]
    ytick_labels = tuple(map(str,ytick_locs))
    im[kidx].set_yticks(ytick_locs)
    im[kidx].set_yticklabels(ytick_labels, color=text_color, fontsize=6, horizontalalignment='right')
    im[kidx].set_ylabel('Mean AUC', fontsize=6, color=text_color)

    im[kidx].axis([0,X_max,0.5,1])
    fig[kidx].suptitle('k = %d (Adaboost)' % (ks[kidx]), x=0.97, y=0.95, color=text_color, fontsize=8, verticalalignment='top', horizontalalignment='right')
    fig[kidx].patch.set_alpha(0)
    im[kidx].patch.set_alpha(1)
    

    leg = im[kidx].legend(loc=4, ncol=2, numpoints=1, labelspacing=0.001, columnspacing=0.1, handlelength=0.5, handletextpad=0.1, frameon=False)
    for l in leg.get_texts():
        l.set_fontsize('5')

    outfile = project_path+'fig/'+virus_family+'_boosterror_%d.pdf' % ks[kidx]
    fig[kidx].savefig(outfile,dpi=DPI,format='pdf')
