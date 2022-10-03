from CODE.Traitement import *
import timeit

import matplotlib.pyplot as plt


def calcul_time_collection():
    start = timeit.default_timer()
    cleaner = Data()
    collection, max_freq_doc = cleaner.clean_data("text.txt")
    end = timeit.default_timer()

    print("the time : ", end - start, " sec")


def calcul_time_fi():
    cleaner = Data()
    collection, max_freq_doc = cleaner.clean_data("text.txt")
    start = timeit.default_timer()
    # index = cleaner.inversed_index_poids(collection, max_freq_doc)
    index = cleaner.File_Invers(collection)
    end = timeit.default_timer()
    print("le nombre de token unique : ", len(index))
    print("la taille de collection : ", len(collection))

    print("the time  fichier inverse : ", end - start, " sec")


import random


def generate_query_randomly(unique_tokens, nbr_var):
    q = ""
    op = ["and", "or", "not"]
    for i in range(nbr_var - 1):
        q += unique_tokens[random.randint(0, len(unique_tokens) - 1)] + op[random.randint(0, 2)]

    return q + unique_tokens[random.randint(0, len(unique_tokens) - 1)]


def evaluate_bool(min, max):
    cleaner = Data()
    collection, max_freq_doc = cleaner.clean_data("text.txt")
    index = cleaner.inversed_index_poids(collection, max_freq_doc)
    unique_tokens = list(set(index.keys()))
    time = []
    nbr_var = []
    # easy ==> var = max 4
    # avg ==> var = max = 15
    # avg ==> var = max = 30
    for i in range(30):
        n = random.randint(min, max)
        nbr_var.append(n)
        query = generate_query_randomly(unique_tokens, n)
        start = timeit.default_timer()
        res = cleaner.bool_model(query, collection, stemming=True)
        end = timeit.default_timer()
        time.append(end - start)
    return nbr_var, time


"""
nbr_var1, time1 = evaluate_bool(2,4)

print("max de var : ", max(nbr_var1))
print("min de var : ", min(nbr_var1))
print("moy de var : ", sum(nbr_var1)/len(nbr_var1))

print("max de temps : ", max(time1))
print("min de temps : ", min(time1))
print("moy de temps : ", sum(time1)/len(time1))


nbr_var2, time2 = evaluate_bool(2,4)

print("max de var : ", max(nbr_var2))
print("min de var : ", min(nbr_var2))
print("moy de var : ", sum(nbr_var2)/len(nbr_var2))

print("max de temps : ", max(time2))
print("min de temps : ", min(time2))
print("moy de temps : ", sum(time2)/len(time2))


nbr_var3, time3 = evaluate_bool(16,30)

print("max de var : ", max(nbr_var3))
print("min de var : ", min(nbr_var3))
print("moy de var : ", sum(nbr_var3)/len(nbr_var3))

print("max de temps : ", max(time3))
print("min de temps : ", min(time3))
print("moy de temps : ", sum(time3)/len(time3))

import matplotlib.pyplot as plt
x = range(1,31)
plt.plot(x, time1, label="c1")

# plotting the line 2 points
plt.plot(x, time2, label="c2")
plt.plot(x, time3, label="c3")

# naming the x axis
plt.xlabel('Requetes - axis')
# naming the y axis
plt.ylabel("Temps d'exéction- axis")
# giving a title to my graph
plt.title('Comparaison - modèle booléen')
# show a legend on the plot
plt.legend()
# function to show the plot
plt.show()
"""


# to get the min nbr doc that maximize the fmeasure
def expeirements(nbr_max, stemming=False, method="PI"):
    print("STEMMING = ", stemming)
    print("RSV METHOD = ", method)

    cleaner = Data()
    if stemming == False:
        collection, max_freq_doc = cleaner.clean_data("text.txt")
        index = cleaner.inversed_index_poids(collection, max_freq_doc)
    else:
        index, collection = cleaner.restore()

    test_q = cleaner.get_test_query()
    qrels = cleaner.get_test_query_res()
    fmeasure = 0
    best_fmeasure = 0
    best_nbr_doc = 0

    y = []

    for nbr_doc in range(1, nbr_max + 1, 1):
        # le vecteur dans ce cas n'est pas trier !
        for qid, res in qrels.items():
            query = test_q[qid]
            model_res = cleaner.evaluate_query_vect_tf(query, index, collection, method, stemming)
            model_res = dict(list(model_res.items())[:nbr_doc])
            fmeasure = fmeasure + cleaner.fmeasure(model_res, res)
        print("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        fmeasure = fmeasure / len(qrels)
        print("FOR NBR DOC == ", nbr_doc)
        print("THE FMEASURE IS == ", fmeasure)

        y.append(fmeasure)

        if best_fmeasure < fmeasure:
            best_fmeasure = fmeasure
            best_nbr_doc = nbr_doc

    print("\n\n-------------------------------------------------------")
    print("THE BEST NBR DOC THAT GIVED THE BEST FMEASURE IS :", best_nbr_doc)
    print("WITH BEST FMEASURE = ", best_fmeasure)

    x = range(1, nbr_max + 1, 1)
    plt.plot(x, y, label="Fmeasure - "+method)

    # naming the x axis
    plt.xlabel('Documents number')
    # naming the y axis
    plt.ylabel("Fmeasure")
    # giving a title to my graph
    plt.title('Fmeasure - Document number')
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.show()
    return y


def expeirements_one_query(nbr_max, query_nbr, stemming=False, method="PI"):
    print("STEMMING = ", stemming)
    print("RSV METHOD = ", method)

    cleaner = Data()
    if stemming == False:
        collection, max_freq_doc = cleaner.clean_data("text.txt")
        index = cleaner.inversed_index_poids(collection, max_freq_doc)
    else:
        index, collection = cleaner.restore()

    test_q = cleaner.get_test_query()[query_nbr]
    qrels = cleaner.get_test_query_res()[query_nbr]

    best_fmeasure = 0
    best_nbr_doc = 0

    y = []

    for nbr_doc in range(1, nbr_max + 1, 1):
        # le vecteur dans ce cas n'est pas trier !

        model_res = cleaner.evaluate_query_vect(test_q, index, collection, method, stemming)
        model_res = dict(list(model_res.items())[:nbr_doc])
        fmeasure = cleaner.fmeasure(model_res, qrels)
        print("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("FOR NBR DOC == ", nbr_doc)
        print("THE FMEASURE IS == ", fmeasure)
        if best_fmeasure < fmeasure:
            best_fmeasure = fmeasure
            best_nbr_doc = nbr_doc

        y.append(fmeasure)


    print("\n\n-------------------------------------------------------")
    print("THE BEST NBR DOC THAT GIVED THE BEST FMEASURE IS :", best_nbr_doc)
    print("WITH BEST FMEASURE = ", best_fmeasure)


    x = range(1, nbr_max + 1, 1)
    plt.plot(x, y, label="Fmeasure")

    # naming the x axis
    plt.xlabel('Documents number')
    # naming the y axis
    plt.ylabel("Fmeasure")
    # giving a title to my graph
    plt.title('Fmeasure - Document number')
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.show()
    return best_fmeasure, best_nbr_doc


def calcul_recall(stemming=False, method="PI"):
    print("STEMMING = ", stemming)
    print("RSV METHOD = ", method)

    cleaner = Data()
    if stemming == False:
        collection, max_freq_doc = cleaner.clean_data("text.txt")
        index = cleaner.inversed_index_poids(collection, max_freq_doc)
    else:
        index, collection = cleaner.restore()

    test_q = cleaner.get_test_query()
    qrels = cleaner.get_test_query_res()
    recall = 0

    y = []

    for qid, res in qrels.items():
        query = test_q[qid]
        model_res = cleaner.evaluate_query_vect(query, index, collection, method, stemming)
        r = cleaner.recall(model_res, res)
        recall = recall + r
        y.append(r)

    print("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    recall = recall / len(qrels)
    print("THE RECALL IS == ", recall)

    x = range(1, len(qrels) + 1)
    plt.plot(x, y, label="Recall")

    # naming the x axis
    plt.xlabel('Query')
    # naming the y axis
    plt.ylabel("Recall")
    # giving a title to my graph
    plt.title('Recall - Query - with Stemming = ' + str(stemming) + ' and method = ' + method)
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.show()


def calcul_precision(stemming=False, method="PI"):
    print("STEMMING = ", stemming)
    print("RSV METHOD = ", method)

    cleaner = Data()
    if stemming == False:
        collection, max_freq_doc = cleaner.clean_data("text.txt")
        index = cleaner.inversed_index_poids(collection, max_freq_doc)
    else:
        index, collection = cleaner.restore()

    test_q = cleaner.get_test_query()
    qrels = cleaner.get_test_query_res()
    precision = 0

    y = []

    for qid, res in qrels.items():
        query = test_q[qid]
        model_res = cleaner.evaluate_query_vect(query, index, collection, method, stemming)
        p = cleaner.precesion(model_res, res)
        precision = precision + p
        y.append(p)

    print("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    precision = precision / len(qrels)
    print("THE PRECISION IS == ", precision)

    x = range(1, len(qrels) + 1)
    plt.plot(x, y, label="precision")

    # naming the x axis
    plt.xlabel('Query')
    # naming the y axis
    plt.ylabel("precision")
    # giving a title to my graph
    plt.title('precision - Query - with Stemming = ' + str(stemming) + ' and method = ' + method)
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.show()


def calcul_fmeasure(stemming=False, method="PI"):
    print("STEMMING = ", stemming)
    print("RSV METHOD = ", method)

    cleaner = Data()
    if stemming == False:
        collection, max_freq_doc = cleaner.clean_data("text.txt")
        index = cleaner.inversed_index_poids(collection, max_freq_doc)
    else:
        index, collection = cleaner.restore()

    test_q = cleaner.get_test_query()
    qrels = cleaner.get_test_query_res()
    fmeasure = 0

    y = []

    for qid, res in qrels.items():
        query = test_q[qid]
        model_res = cleaner.evaluate_query_vect(query, index, collection, method, stemming)
        f = cleaner.fmeasure(model_res, res)
        fmeasure = fmeasure + f
        y.append(f)

    print("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    fmeasure = fmeasure / len(qrels)
    print("THE FMEASURE IS == ", fmeasure)

    x = range(1, len(qrels) + 1)
    plt.plot(x, y, label="fmeasure")

    # naming the x axis
    plt.xlabel('Query')
    # naming the y axis
    plt.ylabel("fmeasure")
    # giving a title to my graph
    plt.title('fmeasure - Query - with Stemming = ' + str(stemming) + ' and method = ' + method)
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.show()




"""

#res de la req 5 :
cleaner = Data()
index, collection = cleaner.restore()
qrels, qtest = cleaner.get_test_query_res(), cleaner.get_test_query()

res = cleaner.evaluate_query_vect(qtest[5],index, collection,"JAC")
print(cleaner.recall(res, qrels[5]))
print(cleaner.precesion(res, qrels[5]))
print(cleaner.fmeasure(res, qrels[5]))
"""



"""

fpi = expeirements(50, stemming=True, method="PI")
fjac = expeirements(50, stemming=True, method="JAC")
fdice= expeirements(50, stemming=True, method="DICE")
fcos= expeirements(50, stemming=True, method="COS")

x = range(1, 50 + 1, 1)
plt.plot(x, fpi, label="Fmeasure- PI")
plt.plot(x, fcos, label="Fmeasure- COS")
plt.plot(x, fdice, label="Fmeasure- DICE")
plt.plot(x, fjac, label="Fmeasure- JAC")

# naming the x axis
plt.xlabel('Documents number')
# naming the y axis
plt.ylabel("Fmeasure")
# giving a title to my graph
plt.title('Fmeasure - Document number')
# show a legend on the plot
plt.legend()
# function to show the plot
plt.show()
"""



"""

flist = []
nlist = []
for i in range(1,21):

    f, nbr = expeirements_one_query(150, i, stemming=True, method="JAC")
    flist.append(f)
    nlist.append(nbr)

print("moy de f ", sum(flist)/len(flist))
print("moy de n ", sum(nlist)/len(nlist))
"""



"""
cleaner = Data()

index, collection = cleaner.restore()
qres = cleaner.get_test_query_res()[5]
q = cleaner.get_test_query()[5]



res = cleaner.evaluate_query_vect(q, index, collection, "JAC", stemming=True)

print(cleaner.recall(res, qres))
print(cleaner.precesion(res, qres))
print(cleaner.fmeasure(res, qres))

# calcul_recall()
"""

"""

calcul_precision(stemming=True)
calcul_precision(method="JAC", stemming=True)
calcul_precision(method="DICE", stemming=True)
calcul_precision(method="COS", stemming=True)
"""


# expeirements(1500)


