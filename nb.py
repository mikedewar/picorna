from nltk.classify import naivebayes as nb
import numpy as np
import cPickle
import splitdata

fh = open('picorna_virii_data_8_2.pkl')
X = cPickle.load(fh)
Y = cPickle.load(fh)
z = cPickle.load(fh)

# just pull out ten features from X to make sure the whole thing works
X = X[:10,:]

v = z.values()
split_indices = splitdata.cv_multiclass_fold(Y,10)

labels = np.argmax(Y,axis=0)
labelled_featuresets = [(dict(zip(v,data)),y) for (data,y) in zip(X.T,labels)]

test_labels=[]
true_labels=[]
for i,train_indices in enumerate(split_indices):
    test_indices = list(set(range(Y.shape[1])).difference(train_indices))
    # train
    train_features = [labelled_featuresets[i] for i in train_indices]
    model = nb.NaiveBayesClassifier.train(train_features)
    # test
    test_features = [labelled_featuresets[i] for i in test_indices]
    label = [model.classify(featureset[0]) for featureset in test_features]
    # collect
    true_labels.append(Y[:,test_indices])
    test_labels.append(label)
# save
fh = open('/proj/ar2384/picorna/labels.pkl','w')
labels = {
    "true":true_labels,
    "test":test_labels
}
cPickle.Pickler(fh,protocol=2).dump(labels)
