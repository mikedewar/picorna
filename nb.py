from nltk.classify import naivebayes as nb
import numpy as np
import cPickle

fh = open('picorna_virii_data_8_2.pkl')
X = cPickle.load(fh)
Y = cPickle.load(fh)
z = cPickle.load(fh)

v = z.values()
labels = np.argmax(Y,axis=0)

labeled_featuresets = [(dict(zip(v,data)),y) for (data,y) in zip(X.T,labels)]

cPickle.dump(labeled_featuresets,file=open('labeled_featuresets.pkl'))

#model = nb.NaiveBayesClassifier.train(labeled_featuresets)

