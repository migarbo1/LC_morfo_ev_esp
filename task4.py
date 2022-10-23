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

def test_other_taggers():
    corpus = load_corpus()
    train, test = utils.get_partitions(corpus, 0.9, True)
    num_of_words = utils.get_num_of_words(10, corpus)

    #brill tagger with HMM and UNIGRAM for various Templates
    result_dict = {}
    baselineModels = ['HMM', 'UNI']
    for baseline in baselineModels:
        for templates in range(1,4):
            brill_acc = utils.brill_tagger(train, test, baseline, templates)
            brill_ic = utils.interval_trust(brill_acc, num_of_words)
            result_dict[(baseline,templates)] = (brill_acc, brill_ic)
    print("Brill tagger:")
    print(result_dict)

    #CRF tagger
    print("CRF tagger:")
    crf_acc = utils.crf_tagger(train, test)
    print(crf_acc, utils.interval_trust(crf_acc, num_of_words))

    #perceptron tagger
    print('Perceptron tagger:')
    perceptron_acc = utils.perceptron_tagger(train, test)
    print(perceptron_acc, utils.interval_trust(perceptron_acc, num_of_words))


test_other_taggers()
