import utils as utils

def test_other_taggers():
    corpus = utils.load_corpus()
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
