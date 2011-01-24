import numpy as np
import pydot as dot
import cPickle
import sys
import pdb

(k,m,fdx,truncate) = sys.argv[1].split(',')
project_path = '/proj/ar2384/picorna/'
virus_family = 'rhabdo'
truncate = int(truncate)

# load in tree from file
f = open(project_path + 'cache/' + virus_family + '_decisiontree_%s_%s_%s.pkl' % (k,m,fdx),'r')
dectree = cPickle.load(f)
order = cPickle.load(f)
f.close()
order.insert(0,-1)
order = order[:truncate]

"""
# load kmer dictionary
f = open('../data/picorna_virii_data_%s_%s.pkl' % (k,m),'r')
X = cPickle.load(f)
Y = cPickle.load(f)
kmer_dict = cPickle.load(f)
f.close()
"""

# initialize graph
graph = dot.Dot(graph_type='digraph',size='9.')

# initialize nodes
nodes = dict()
for o in order:
#    k_id = dectree[o][0]

    # add a node for the k-mer to the dictionary of nodes
    if o==-1:
        nodename = 'Root'
    else:
#        kmer = kmer_dict[k_id]
        kmer = dectree[o][0][0]
        threshold = dectree[o][0][1]
        nodename = '(%d) [|%s| < %d] ?' % (o,kmer,threshold)
    nodes[o] = [dot.Node(nodename,shape='box',fontsize='8.',fontcolor='blue')]

    # add a node for the decision output {True,False} to
    # the dictionary of nodes
    dtext = ['True','False']
    colors = ['darkgreen','red']
    for i in range(len(dectree[o])-1):
        nodename = dtext[i]+'\\n %.3f' % dectree[o][i+1][0]
        nodes[o].append(dot.Node(nodename,fontsize='6.',fontcolor=colors[i]))

# add nodes to graph
for o in order:
    for i in range(len(nodes[o])):
        graph.add_node(nodes[o][i])
    
# add edges to graph
for o in order:
    for i in range(1,len(nodes[o])):
        # Edge from kmer to decision
        graph.add_edge(dot.Edge(nodes[o][0],nodes[o][i]))
        children = dectree[o][i][1]
        tchildren = children[:]
        for child in tchildren:
            if child not in order:
                children.remove(child)
        if len(children):
            # Edge from decision to child k-mers
            for child in children:
                graph.add_edge(dot.Edge(nodes[o][i],nodes[child][0]))

graph.write_jpeg(project_path + 'fig/' + virus_family + '_decisiontree_%s_%s_%s.jpg' % (k,m,fdx))
