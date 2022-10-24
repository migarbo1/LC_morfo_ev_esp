import utils as utils

def tnt_smoothing_test():
    corpus = utils.load_corpus()
    acc_array = utils.fold_cross_validation(corpus, 10, True, 'TNT')
    n = utils.get_num_of_words(10, corpus)
    ic = [utils.interval_trust(acc,n) for acc in acc_array]
    utils.plot_model(results=acc_array, ic=ic, fold=10, model="TNT")
    print("No smoothing")
    for i in range(len(acc_array)):
        print(str(acc_array[i]) + " +- " + str(ic[i]))

    for suffix_size in range(1, 5):
        acc_array = utils.fold_cross_validation(corpus, 10, True, 'TNT', -1*suffix_size)
        ic = [utils.interval_trust(acc,n) for acc in acc_array]
        utils.plot_model(results=acc_array, ic=ic, fold=10, model='TNT')
        print("Affix length : " + str(suffix_size))
        for i in range(len(acc_array)):
            print(str(acc_array[i]) + " +- " + str(ic[i]))
    


tnt_smoothing_test()
