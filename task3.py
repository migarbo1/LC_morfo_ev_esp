import nltk
import random
import numpy as np
from nltk.corpus import cess_esp
import utils as utils

def load_corpus():
    corpus_sentences=list(cess_esp.tagged_sents())

    norm_corpus = []
    for sentence in corpus_sentences:
        norm_sentence = []
        for w,cat in sentence:
            if w != '*0*':
                if str(cat).startswith('v'):
                    cat = cat[0:3] if len(cat) >=3 else cat
                elif str(cat).startswith('F'):
                    cat = cat[0:3] if len(cat) >=3 else cat
                else:
                    cat = cat[0:2] if len(cat) >=3 else cat
                norm_sentence.append((w,cat))
        norm_corpus.append(norm_sentence)
    return norm_corpus

def tnt_smoothing_test():
    corpus = load_corpus()
    acc_array = utils.fold_cross_validation(corpus, 10, True, 'TNT')
    n = utils.get_num_of_words(10, corpus)
    ic = [utils.interval_trust(acc,n) for acc in acc_array]
    utils.plot_model(results=acc_array, ic=ic, fold=10, model="HMM")
    print("No smoothing")
    for i in range(len(acc_array)):
        print(str(acc_array[i]) + " +- " + str(ic[i]))

    for suffix_size in range(1, 5):
        acc_array = utils.fold_cross_validation(corpus, 10, True, 'TNT', suffix_size)
        ic = [utils.interval_trust(acc,n) for acc in acc_array]
        utils.plot_model(results=acc_array, ic=ic, fold=10, model='TNT')
        print("Affix length : " + str(suffix_size))
        for i in range(len(acc_array)):
            print(str(acc_array[i]) + " +- " + str(ic[i]))
    


tnt_smoothing_test()
