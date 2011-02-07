import numpy as np
import cPickle
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import sys
import pdb

def represent_sequence(sequence, k):

    """
    Convert sequence into vector of kmers.
    """

    # process sequence
    sequence = list(sequence)
    seq_len = len(sequence)
    kmers = []
    for col in range(k):
        kmers.append(list(sequence[col:seq_len-k+col+1]))
    kmers = np.array(kmers)

    return kmers

def compute_prediction(kmers, k, m, folds):

    """
    Compute predictions for a vector of kmers.
    """

    predictions = []
    selected_kmer = []
    node_outputs = dict()
    for fold in range(folds):
        # load decision tree
        try:
            f = open(datapath + virus_family + '_decisiontree_%d_%d_%d.pkl' % (k,m,fold), 'r')
        except IOError:
            continue
        decision_tree = cPickle.load(f)
        order = cPickle.load(f)
        f.close()
        order.insert(0,-1)

        prediction = decision_tree[-1][1][0] * decision_tree[-1][1][2]
        node_outputs[-1] = [1]
        parent_found = False
        for node in order[1:cut_off]:
            node_index = order.index(node)
            # find parent
            for p in order[:node_index]:
                choices = range(len(decision_tree[p])-1)
                for choice in choices:
                    if node in decision_tree[p][choice+1][1]:
                        parent = (p,choice)
                        parent_found = True
                if parent_found:
                    break
            (kmer, threshold) = decision_tree[node][0]
            selected_kmer.append(list(kmer))

            # COMPUTE COUNTS
            kmer = np.array(list(kmer)).reshape(k,1)
            count = ((kmer != kmers).sum(0)<=m).sum()

            parent_output = node_outputs[parent[0]][parent[1]]
            node_outputs[node] = [(count<threshold) * parent_output, (count>=threshold) * parent_output]
            prediction += node_outputs[node][0] * decision_tree[node][1][0] * decision_tree[node][1][2] + node_outputs[node][1] * decision_tree[node][2][0] * decision_tree[node][2][2]
        predictions.append(prediction)

    predictions = np.array(predictions).reshape(folds,2)

    return predictions

def plot_prediction_histogram(prediction, labels, validation, threshold, k, m):
    """
    Plot histogram of predictions and mark
    validation data in the histogram.
    """

    bg_color = 'w'
    text_color = 'k'

    DPI = 300 
    fig_resolution = (1024,768)
    fig_size = tuple([res/float(DPI) for res in fig_resolution])
    fig = plot.figure(figsize = fig_size, facecolor = bg_color, edgecolor = bg_color)
    im = fig.add_subplot(111, axisbg=bg_color)

    x_min = np.floor(min([prediction.min(), validation.min()]))
    x_max = np.ceil(max([prediction.max(), validation.max()]))
    y_max = 8
    bins = [x*0.2 for x in range(x_min*5, x_max*5+0.2)]

    im.hist([prediction[(labels>0).nonzero()[0]], prediction[(labels<0).nonzero()[0]]], bins=bins, color=('blue','red'), edgecolor='none', label=['Plants','Animals'])
    im.scatter(validation, [4.,4.5], s=10, facecolors='k', edgecolor='none', marker='o')
    im.text(validation[0]+0.1, 4., 'taastrup', fontsize=6, color='k')
    im.text(validation[1]+0.1, 4.5, 'farmington', fontsize=6, color='k')
#    im.annotate('taastrup', xy=(validation[0], 0),  xycoords='data', xytext=(0, 30), textcoords='offset points', arrowprops=dict(arrowstyle="->"), fontsize=10, color=text_color)
#    im.annotate('farmington', xy=(validation[1], 0),  xycoords='data', xytext=(10, 20), textcoords='offset points', arrowprops=dict(arrowstyle="->"), fontsize=10, color=text_color)
#    im.plot(np.ones(2)*threshold, [0,5], '-c', linewidth=1)

    x_min = np.floor(min([prediction.min(), validation.min()]))
    x_max = np.ceil(max([prediction.max(), validation.max()]))
    y_max = 20

    im.axis([x_min, x_max, 0, y_max])

    xtick_locs = range(x_min, x_max+1)
    xtick_labels = tuple(map(str,xtick_locs))
    im.set_xticks(xtick_locs)
    im.set_xticklabels(xtick_labels, color=text_color, fontsize=6, verticalalignment='center')
    im.set_xlabel('Predictions', fontsize=6, color=text_color)

    ytick_locs = range(0,y_max,4)
    ytick_labels = tuple(map(str,ytick_locs))
    im.set_yticks(ytick_locs)
    im.set_yticklabels(ytick_labels, color=text_color, fontsize=6, horizontalalignment='right')
 
    fig.suptitle('k = %d, m = %d' % (k,m), x=0.8, y=0.95, color=text_color, fontsize=8, verticalalignment='top', horizontalalignment='right')
#    fig[kidx].patch.set_alpha(0)
#    im[kidx].patch.set_alpha(1)

    leg = im.legend(loc=1, numpoints=1, labelspacing=0.001, handlelength=0.5, handletextpad=0.1, frameon=False)
    for l in leg.get_texts():
        l.set_fontsize('5')

    outfile = project_path+'fig/'+virus_family+'_prediction_histogram_%d_%d.pdf' % (k,m)
    fig.savefig(outfile,dpi=DPI,format='pdf')


if __name__=="__main__":

    project_path = '/proj/ar2384/picorna/'
    (k, m, cut_off) = map(int,sys.argv[1:])
    folds = 10

    sequences = sys.stdin.readlines()

    datapath = '/proj/ar2384/picorna/cache/'
    virus_family = 'rhabdo'

    kmers = [represent_sequence(sequence, k) for sequence in sequences]

    validations = [compute_prediction(kmer_vec, k, m, folds) for kmer_vec in kmers]
    validation = np.array(validations).reshape(len(sequences), folds, 2)

    f = open(datapath + virus_family + '_virii_test_output_%d_%d.pkl' % (k,m), 'r')
    Fidx = cPickle.load(f)
    f.close()

    f = open(datapath + virus_family + '_virii_data_%d_%d.pkl' % (k,m), 'r')
    Xt = cPickle.load(f)
    Yt = cPickle.load(f)
    f.close()
    (Dt,Nt) = Xt.shape
    Kt = Yt.shape[0]

    predictions = np.empty((Kt,0))
    labels = np.empty((Kt,0))
    thresholds = np.zeros(folds)
    for fold in range(folds):

#        test_indices = Fidx[fold]
#        test_indices.sort()
#        indices = list(set(range(Nt)).difference(set(test_indices)))
#        indices.sort()
#        indices.extend(test_indices)
        indices = Fidx[fold]
        indices.sort()
        labels = np.hstack((labels, Yt[:,indices]))
 
        # load learning algorithm output
        f = open(datapath + virus_family + '_dumptree_%d_%d_%d.pkl' % (k,m,fold), 'r')
        train_pred = cPickle.load(f)
        test_pred = cPickle.load(f)
        rocacc = cPickle.load(f)
        f.close()
        thresholds[fold] = rocacc[cut_off,-1]

#        predictions = np.hstack((predictions, train_pred[:,:,cut_off]))
        predictions = np.hstack((predictions, test_pred[:,:,cut_off]))

    plot_prediction_histogram(predictions[0,:], labels[0,:], validation[:,:,0].mean(1), thresholds.mean(), k, m)
