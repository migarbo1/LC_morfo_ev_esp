#import zone
import numpy as np
import math
import random
from nltk.tag import hmm
from nltk.tag import tnt
import matplotlib.pyplot as plt

#useful functions
def cosinus_distance(e1, e2):
    '''
    function to compute the cosinusdistance between two word embeddings
    '''
    return np.dot(e1, e2)/(np.sqrt(e1.dot(e1))*np.sqrt(e2.dot(e2)))

def interval_trust(P, N):
    '''
    function to compute the trust interval for a given accuracy(N) and test size(N)
    '''
    return 1.96*math.sqrt((P*(1-P))/N)

def hmm_tagger(train, test):
    '''
    function to train and test a HMM model
    '''
    local_hmm_tagger = hmm.HiddenMarkovModelTagger.train(train)
    hmm_tagger_accuracy = local_hmm_tagger.accuracy(test)
    return hmm_tagger_accuracy

def tnt_tagger(train, test):
    '''
    function to train and test a TNT model
    '''
    local_tnt_tagger = tnt.TnT()
    local_tnt_tagger.train(train)
    tnt_tagger_accuracy = local_tnt_tagger.accuracy(test)
    return tnt_tagger_accuracy

def fold_cross_validation(corpus, fold=10, shuffle=False, model = 'Both'):
    '''
    function to compute N-fold cross validation over a given corpus with or withour shuffle. 
    model can have the following values: HMM, TNT, Both
    '''
    fold_cross_results = []
    local_corpus = corpus.copy()
    if shuffle:
        random.shuffle(local_corpus)
    len_subsets = math.floor(len(corpus)/fold)

    for i in range(fold):
        start = i*len_subsets
        end = (i+1)*len_subsets if i < fold-1 else -1
        _test = local_corpus[start:end]
        if(start != 0 and end != -1):
            _train = local_corpus[0:start] + local_corpus[end:]
        elif start == 0:
            _train = local_corpus[end:]
        elif end == -1:
            _train = local_corpus[0:start]
        
        if model == 'HMM':
            prec_hmm = hmm_tagger(_train, _test)
            fold_cross_results.append(prec_hmm)
        if model == 'TNT':
            prec_tnt = tnt_tagger(_train, _test)
            fold_cross_results.append(prec_tnt)
        if model == 'Both':
            prec_hmm = hmm_tagger(_train, _test)
            prec_tnt = tnt_tagger(_train, _test)
            fold_cross_results.append(np.array([prec_hmm, prec_tnt]))

    return np.array(fold_cross_results)

def get_num_of_words(fold, corpus):
    n = 0
    for sentence in corpus:
        for word in sentence:
            n += 1
    return n//fold

def plot_model(results, ic, model, fold):
    x=[i+1 for i in range(fold)]
    y=results
    #plt.axis([0, 11, 0.80, 0.97])
    plt.ylabel('Accuracy')
    plt.xlabel('Fold')
    plt.title('Ten-fold cross validation ' + model)
    plt.plot(x,y,'ro')
    plt.errorbar(x,y,yerr=ic,linestyle='None')
    plt.margins(y=0.5)
    plt.show()
