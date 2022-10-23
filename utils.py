#import zone
import numpy as np
import math
import random
import nltk
from nltk.tag import hmm, tnt, brill, brill_trainer, UnigramTagger, crf, perceptron
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

def tnt_tagger(train, test, smoothing):
    '''
    function to train and test a TNT model
    '''
    if smoothing > 0 :
        local_affix_tagger = affix_tagger(train, affix_len=smoothing)
        local_tnt_tagger = tnt.TnT(unk=local_affix_tagger, Trained=True)
    else:
        local_tnt_tagger = tnt.TnT()
    local_tnt_tagger.train(train)
    tnt_tagger_accuracy = local_tnt_tagger.accuracy(test)
    return tnt_tagger_accuracy

def affix_tagger(train, test=None, affix_len = 3):
    local_affix_tagger = nltk.AffixTagger(train, affix_length=affix_len)
    if test == None:
        return local_affix_tagger
    else:
        affix_tagger_accuracy = local_affix_tagger.evaluate(test)
        return affix_tagger_accuracy

def brill_tagger(train, test, baselineModel, templates):
    '''
    function to train and test a brill tagger. 
    - for the baselineModels user can choose: 'UNI' or 'HMM'
    - for the templates user can choose:
        1: demo18 -- templates for the original nltk demo
        2: demo18 plus -- original nltk demo + multi-feature
        3: brill24 -- paper of 1995 
    '''
    selected_template = brill.nltkdemo18()
    if templates == 2:
        selected_template = brill.nltkdemo18plus()
    if templates == 3:
        selected_template = brill.brill24()

    baseline = UnigramTagger(train) if baselineModel == 'UNI' else hmm.HiddenMarkovModelTagger.train(train)
    
    local_bill_tagger = brill_trainer.BrillTaggerTrainer(baseline, selected_template).train(train)
    return local_bill_tagger.evaluate(test)

def crf_tagger(train, test):
    local_crf_tagger = crf.CRFTagger()
    local_crf_tagger.train(train, 'model.crf.tagger')
    return local_crf_tagger.accuracy(test)

def perceptron_tagger(train, test):
    local_perceptron_tagger = perceptron.PerceptronTagger()
    local_perceptron_tagger.train(train)
    return local_perceptron_tagger.accuracy(test)

def fold_cross_validation(corpus, fold=10, shuffle=False, model = 'Both', smoothing=0):
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
            prec_tnt = tnt_tagger(_train, _test, smoothing)
            fold_cross_results.append(prec_tnt)
        if model == 'Both':
            prec_hmm = hmm_tagger(_train, _test)
            prec_tnt = tnt_tagger(_train, _test)
            fold_cross_results.append(np.array([prec_hmm, prec_tnt]))

    return np.array(fold_cross_results)

def incremental_test(corpus, fold=10, shuffle=False, model = 'Both'):
    '''
    function to study accuracy of a model in relation to the size of the corpus. 
    model can have the following values: HMM, TNT, Both
    '''
    results = []
    local_corpus = corpus.copy()
    if shuffle:
        random.shuffle(local_corpus)
    len_subsets = math.floor(len(corpus)/fold)
    
    test = local_corpus[9*len_subsets:]

    for i in range(fold):
        train = local_corpus[:(i+1)*len_subsets]
        
        if model == 'HMM':
            prec_hmm = hmm_tagger(train, test)
            results.append(prec_hmm)
        if model == 'TNT':
            prec_tnt = tnt_tagger(train, test)
            results.append(prec_tnt)
        if model == 'Both':
            prec_hmm = hmm_tagger(train, test)
            prec_tnt = tnt_tagger(train, test)
            results.append(np.array([prec_hmm, prec_tnt]))

    return np.array(results)

def get_partitions(corpus, relation=0.9, shuffle=False):
    '''
    Function that returns train and test partitions based on the relation of their sizes
    returns train, test
    '''
    local_corpus = corpus.copy()

    if shuffle:
        random.shuffle(local_corpus)
    len_train = math.floor(len(local_corpus)*relation)
    train = local_corpus[:len_train]
    test = local_corpus[len_train:]

    return train, test

def get_num_of_words(fold, corpus):
    n = 0
    for sentence in corpus:
        for word in sentence:
            n += 1
    return n//fold

def plot_model(results, ic, model, fold, title='Ten-fold cross validation'):
    x=[i+1 for i in range(fold)]
    y=results
    #plt.axis([0, 11, 0.80, 0.97])
    plt.ylabel('Accuracy')
    plt.xlabel('Fold')
    plt.title('{} {}'.format(title, model))
    plt.plot(x,y,'ro')
    plt.errorbar(x,y,yerr=ic,linestyle='None')
    plt.margins(y=0.1)
    plt.show()
