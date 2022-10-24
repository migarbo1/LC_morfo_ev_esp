import utils as utils

def cessEsp_incremental_Hmm():
    corpus = utils.load_corpus()
    acc_array = utils.incremental_test(corpus, 10, True, 'HMM')
    n = utils.get_num_of_words(10, corpus)
    ic = [utils.interval_trust(acc,n) for acc in acc_array]
    utils.plot_model(results=acc_array, ic=ic, fold=9, title='Incremental validation', model="HMM")
    for i in range(len(acc_array)):
        print(str(acc_array[i]) + " +- " + str(ic[i]))


cessEsp_incremental_Hmm()
