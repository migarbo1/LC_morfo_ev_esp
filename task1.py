import utils as utils

def cessEsp_10Fold_Hmm():
    for value in [True, False]:
        corpus = utils.load_corpus(value)
        acc_array = utils.fold_cross_validation(corpus, 10, True, 'HMM')
        n = utils.get_num_of_words(10, corpus)
        ic = [utils.interval_trust(acc,n) for acc in acc_array]
        utils.plot_model(results=acc_array, ic=ic, fold=10, model="HMM")
        print('reduce corpus tags: {}'.format(value))
        for i in range(len(acc_array)):
            print(str(acc_array[i]) + " +_ " + str(ic[i]))


cessEsp_10Fold_Hmm()
