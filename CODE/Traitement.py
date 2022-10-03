from nltk import word_tokenize, ISRIStemmer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import ISRIStemmer
import re
import numpy as np
import math
import pickle
import json


class Document:
    def __init__(self, I):
        self.I = I.strip()
        self.A = ""
        self.T = ""
        self.W = ""

    def add_T(self, T):
        self.T = T.strip()

    def add_W(self, W):
        self.W = W.strip()

    def add_A(self, A):
        self.A = A.strip()

    def get_A(self):
        return self.A

    def get_I(self):
        return self.I

    def clean_doc(self, stoplist, stemming=False):
        doc = word_tokenize(self.A) + word_tokenize(self.T) + word_tokenize(self.W)
        freq = {}
        max_doc_freq = 0
        for w in doc:
            w = w.strip().lower()
            # if w not in stoplist and len(w) > 0 and (w.isalpha() or w.isnumeric()):
            if w not in stoplist and len(w) > 0 and (w.isalpha()):
                if stemming:
                    # stemmer = SnowballStemmer("english")
                    stemmer = PorterStemmer()
                    # stemmer = ISRIStemmer()
                    w = stemmer.stem(w)
                if freq.__contains__(w):
                    freq[w] += 1
                else:
                    freq[w] = 1

                if freq[w] > max_doc_freq:
                    max_doc_freq = freq[w]

        return freq, max_doc_freq

    def show(self):
        print('----------------------------------\n----------------------------------')
        print('\tattribute I : ')
        print(self.I)
        print('\tattribute T : ')
        print(self.T)
        print('\tattribute W : ')
        print(self.W)
        print('\tattribute A : ')
        print(self.A)

    def printf(self):
        print(self.I, '\n', self.A, self.T, self.W)


class Data:

    # def __init__(self, path="CACM"):
    def __init__(self, path="..\\CACM"):

        self.dir_path = path
        self.collection = []

    def download(self, nomFile):
        data = []
        file = open(self.dir_path + "\\" + nomFile, encoding='utf-8').read()
        liste = re.findall(r"([^#]*)", file)
        liste = [l.strip() for l in liste if len(l.strip()) > 0]
        i = 0
        while i < len(liste) - 1:
            # print(i)
            # print(liste[i])
            if liste[i].startswith('D'):
                if i != 0:
                    data.append(obj)
                obj = Document(liste[i])
                i += 1
            elif liste[i] == "Authors":
                obj.add_A(liste[i + 1])
                i += 2
            elif liste[i] == "Summary":
                obj.add_W(liste[i + 1])
                i += 2
            elif liste[i] == "Titles":
                obj.add_T(liste[i + 1])
                i += 2
            elif liste[i] == "Ignore":
                i += 2
        data.append(obj)
        return data



    def clean_data(self, nomeFile):
        # cleaning all the attributes in one doc :
        cleaned_collection = {}
        self.collection = self.download(nomeFile)
        docs_max_freq = {}
        common_words = open(self.dir_path + '\\common_words.txt')
        stoplist = stopwords.words('english')
        stoplist = stoplist + [w.strip().lower() for w in common_words]
        stoplist = set(stoplist)

        #stoplist = [w.strip().lower() for w in common_words]
        for doc in self.collection:
            clean, max_freq = doc.clean_doc(stoplist)
            cleaned_collection[doc.get_I()] = clean
            docs_max_freq[doc.get_I()] = max_freq

        return cleaned_collection, docs_max_freq


    """
    
    def clean_data(self, stemming=True):
        # cleaning all the attributes in one doc :
        cleaned_collection = {}
        self.collection = self.download()
        docs_max_freq = {}
        common_words = open(self.dir_path + '\\common_words.txt')
        stoplist = stopwords.words('english')
        stoplist = stoplist + [w.strip().lower() for w in common_words]
        stoplist = set(stoplist)

        for doc in self.collection:
            clean, max_freq = doc.clean_doc(stoplist, stemming)
            cleaned_collection[doc.get_I()] = clean
            docs_max_freq[doc.get_I()] = max_freq

        return cleaned_collection, docs_max_freq
    """

    def File_Invers(self, collection):
        dic = {}
        # i = 0
        for doc, terms in collection.items():
            # print(doc, terms)
            for term, freq in terms.items():
                if term not in dic:

                    dic[term] = {doc: freq}
                    # print(dic)
                else:
                    dic[term].update({doc: freq})
        return dic

    def ni(self, collection, terme):
        ni = 0
        for _, v in collection.items():
            if terme in v:
                ni += 1
        return ni

    def inversed_index_poids(self, collection, max_freq):
        poids = {}

        for k, v in collection.items():
            docs = {}
            for t in v.keys():
                if t not in poids:
                    poids[t] = {}

                # poids[t][k] = "%.2f" % ((v[t] / max_freq[k]) * (np.log(len(collection) / self.ni(collection, t) + 1)))
                poids[t][k] = round((v[t] / max_freq[k]) * (np.log(len(collection) / self.ni(collection, t) + 1)), 3)

        return poids

    def RSV_COS(self, doc_name, terms, query, index, collection):
        pd = 0
        w_doc = 0
        w_query = 0
        for tq, _ in query.items():
            if tq in terms:
                pd = pd + index[tq][doc_name] * query[tq]
            # dans les tout cas je fait la somme !
            w_query = w_query + math.pow(query[tq], 2)

        """
        
        for _, freq in collection[doc_name].items():
            w_doc = w_doc + math.pow(freq, 2)
        
        """
        for token, _ in terms.items():
            w_doc = w_doc + math.pow(index[token][doc_name],2)


        if (math.sqrt(w_query * w_doc)) != 0:
            return pd / (math.sqrt(w_query * w_doc))
        else:
            return 0.0


    def RSV_DICE(self, doc_name, terms, query, index, collection):
        pd = 0
        w_doc = 0
        w_query = 0
        for tq, _ in query.items():
            if tq in terms:
                pd = pd + index[tq][doc_name] * query[tq]
            w_query = w_query + math.pow(query[tq], 2)
        """
        for _, freq in collection[doc_name].items():
            w_doc = w_doc + math.pow(freq, 2)
        """

        for token, _ in terms.items():
            w_doc = w_doc + math.pow(index[token][doc_name],2)

        if (w_query + w_doc != 0):
            return 2 * pd / (w_query + w_doc)
        else:
            return 0.0

    def RSV_JAC(self, doc_name, terms, query, index, collection):
        pd = 0
        w_doc = 0
        w_query = 0
        # for t, freq in terms.items():
        for tq, _ in query.items():
            if tq in terms:
                pd = pd + index[tq][doc_name] * query[tq]
            w_query = w_query + math.pow(query[tq], 2)
        """
        
        for _, freq in collection[doc_name].items():
            w_doc = w_doc + math.pow(freq, 2)
        """
        for token, _ in terms.items():
            w_doc = w_doc + math.pow(index[token][doc_name], 2)

        if (w_query + w_doc - pd) != 0:
            return pd / (w_query + w_doc - pd)
        else:
            return 0.0

    def read_query_vect(self, query_text, unique_tokens, stemming):
        # dict key = term , value = 1
        query = {}

        #print(query_text)
        for w in query_text.strip().split(' '):

            if stemming:
                stemmer = PorterStemmer()
                w = stemmer.stem(w)
            w = w.lower()
            
            if w in unique_tokens:
                query[w] = 1
            #query[w] = 1
        #print("query read ",len(query))
        return query

    def RSV_PI(self, doc_name, terms, query, index):
        rsv = 0
        # print("query" , query)
        for tq, freq in query.items():
            # for t, _ in terms.items():
            if tq in terms:
                # print("the tq ", tq)
                # print("the query[t] ", query[tq])
                # print("the weight ", index[tq][doc_name])
                rsv = rsv + index[tq][doc_name] * freq
        # print("rsv",rsv)
        return rsv

    def evaluate_query_vect(self, query_text, index, cleaned_collection, method, stemming=True):
        # index = dict key = term , value = dict ( key = doc, value= freq)

        query = self.read_query_vect(query_text, list(index.keys()), stemming)
        # print("query11 ",list(set(index.keys())))
        evaluation = {}  # dict (key = doc , value = RSV)
        #print(method)

        for doc, terms in cleaned_collection.items():

            if method == "PI":
                eval = self.RSV_PI(doc, terms, query, index)
            # print("val" ,eval)

            elif method == "COS":
                eval = self.RSV_COS(doc, terms, query, index, cleaned_collection)
                # print("eval ",eval)

            elif method == "DICE":
                eval = self.RSV_DICE(doc, terms, query, index, cleaned_collection)

            elif method == "JAC":
                eval = self.RSV_JAC(doc, terms, query, index, cleaned_collection)

            else:
                print("ERROR NON VALID METHOD !")

            if eval != 0:
                evaluation[doc] = eval

        #return evaluation
        return dict(sorted(evaluation.items(), key=lambda x: x[1], reverse=True))

        ##########################################################################

    def evaluate_query_vect_tf(self, query_text, index, cleaned_collection, method, stemming=True):
        # index = dict key = term , value = dict ( key = doc, value= freq)

        query = self.read_query_tf(query_text, stemming)
        # print("query11 ",list(set(index.keys())))
        evaluation = {}  # dict (key = doc , value = RSV)
        #print(method)

        for doc, terms in cleaned_collection.items():

            if method == "PI":
                eval = self.RSV_PI(doc, terms, query, index)
            # print("val" ,eval)

            elif method == "COS":
                eval = self.RSV_COS(doc, terms, query, index, cleaned_collection)
                # print("eval ",eval)

            elif method == "DICE":
                eval = self.RSV_DICE(doc, terms, query, index, cleaned_collection)

            elif method == "JAC":
                eval = self.RSV_JAC(doc, terms, query, index, cleaned_collection)

            else:
                print("ERROR NON VALID METHOD !")

            if eval != 0:
                evaluation[doc] = eval

        #return evaluation
        return dict(sorted(evaluation.items(), key=lambda x: x[1], reverse=True))




    def getDocsWithTerme(self, terme, nomeFile):

        collection, _ = self.clean_data(nomeFile)
        docs = []
        for k, v in collection.items():
            if terme in v:
                docs.append(k)
        return docs

    def getDocsTerme(self, doc, nomeFile):
        collection, _ = self.clean_data(nomeFile)
        return collection[doc]

    def getDocsWithTerme1(self, collection, terme):
        docs = []
        for k, v in collection.items():
            if terme in v:
                docs.append(k)
        return docs

    def bool_model(self, query, collection, stemming=True):
        """the query mustn't contain duplicate termes"""
        termes = word_tokenize(query.lower())

        ignore = ['and', 'or', 'not', '(', ')']
        exist = {}
        #####################################################
        # i added this one !
        if stemming:
            stemmer = PorterStemmer()
            termes = [stemmer.stem(w) for w in termes if termes not in ignore]
        ###################################################""
        # svc contain the docs that contain the terme

        for terme in termes:
            if terme not in ignore:
                exist[terme] = self.getDocsWithTerme1(collection, terme)

        # for each docs
        reqToEval = {}
        for k in list(collection.keys()):
            for terme, docs in exist.items():
                if k not in reqToEval:
                    reqToEval[k] = {}
                if k in docs:
                    reqToEval[k][terme] = '1'
                    print()
                else:
                    reqToEval[k][terme] = '0'
        output = []
        for doc, terme in reqToEval.items():
            termesReq = " ".join(termes)
            for t in terme:
                termesReq = termesReq.replace(t, reqToEval[doc][t])
            if eval(termesReq) == 1:
                output.append(doc)
        print(output)
        return output

    def store(self, index, collection):
        index_file = open("index.json", "w")
        collection_file = open("collection.json", "w")

        json.dump(index, index_file)

        json.dump(collection, collection_file)

    def restore(self):
        index_file = open("..\\DATA\\index.json", "r")
        collection_file = open("..\\DATA\\collection.json", "r")

        index = json.load(index_file)
        collection = json.load(collection_file)
        # collection = collection_file.read()

        return index, collection

    def evaluation_Precision(self, vectorResult, numQuery, vectorQrels):
        Res = []
        liste = []
        for elt in vectorResult.keys():
            Res.append(elt[1:])
        for i in Res:
            if i in vectorQrels[numQuery]:
                liste.append(i)
        if (len(Res) != 0):
            Precision = (len(liste) / len(Res))
        else:
            Precision = 'aucun document retourné'

        return Precision

    def evaluation_Rappel(self, vectorResult, numQuery, vectorQrels):
        Res = []
        liste = []
        for elt in vectorResult.keys():
            Res.append(elt[1:])

        for i in Res:
            if i in vectorQrels[numQuery]:
                liste.append(i)

        if len(Res) != 0:
            Precision = (len(liste) / len(Res))
        else:
            Precision = 0

        if (len(vectorQrels[numQuery]) != 0):
            Rappel = (len(liste) / len(vectorQrels[numQuery]))
        else:
            Rappel = 0

        return Rappel, Precision

    def recall(self,model_res, real_res):
        # model_res = dict (key = word, value = sim)
        # real_res = list of real pertinents docs
        nbr = 0
        for doc in model_res.keys():
            doc = int(doc[1:])
            if doc in real_res:
                nbr += 1

        return nbr / len(real_res)

    def precesion(self,model_res, real_res):
        # model_res = dict (key = word, value = sim)
        # real_res = list of real pertinents docs
        nbr = 0
        for doc in model_res.keys():
            doc = int(doc[1:])
            if doc in real_res:
                nbr += 1

        if len(model_res) == 0:
            return 0

        return nbr / len(model_res)

    def fmeasure(self, model_res, real_res):

        r = self.recall(model_res, real_res)
        p = self.precesion(model_res, real_res)

        if r + p !=0:
            return (2 * p * r) / (r + p)
        return 0

    def getqrels(self, file_name='qrels.text'):
        # f = open(file_name, 'r')
        f = open(self.dir_path + "\\" + file_name, 'r')
        text = f.readlines()
        dic_All = {}
        for line in text:
            #     print(line)
            Tab = line.split()
            if Tab[0] not in dic_All:

                dic_All.setdefault(Tab[0], []).append(Tab[1])

            else:
                dic_All.setdefault(Tab[0], []).append(Tab[1])

        return dic_All

    def get_test_query(self):
        query = {}
        q = ""
        q_nbr = 0
        query_file = open(self.dir_path + '\\query.text', encoding='utf-8').read()
        liste = re.findall(r"([^#]*)", query_file)
        liste = [l.strip() for l in liste if len(l.strip()) > 0]
        i = 0
        while i < len(liste) - 1:
            #print(i)
            #print(liste[i])
            if liste[i].startswith('D'):
                if i != 0:
                    query[q_nbr] = q
                q_nbr += 1
                q = ""
                i += 1
            elif liste[i] == "Authors":
                q += " "+liste[i + 1]
                i += 2
            elif liste[i] == "Titles":
                q += " "+liste[i + 1]
                i += 2
            elif liste[i] == "Ignore":
                i += 2
        #q_nbr += 1
        query[q_nbr] = q
        return query

    def get_test_query_res(self):
        res = {}
        query_res_file = open(self.dir_path + '\\qrels.text', encoding='utf-8')
        for line in query_res_file:
            nbrs = [int(n) for n in line.strip().split()]
            if res.__contains__(nbrs[0]):
                res[nbrs[0]].append(nbrs[1])
            else:
                res[nbrs[0]] = [nbrs[1]]

        return res

    def test_query_collection(self, stemming=False):

        q_test = self.get_test_query()

        common_words = open(self.dir_path + '\\common_words.txt')
        stoplist = stopwords.words('english')
        stoplist = stoplist + [w.strip().lower() for w in common_words]
        stoplist = set(stoplist)

        query = {}

        for qid, q_text in q_test.items():
            if stemming :
                stemmer = PorterStemmer()
                q_tokens = [stemmer.stem(q) for q in q_text.split(" ") if len(q) > 0 and q.isalpha() and q not in stoplist]
            else:
                q_tokens= [q for q in q_text.split(" ") if len(q) > 0 and q.isalpha() and q not in stoplist]

            q_freq = {}
            for token in q_tokens:
                if token in q_freq.keys():
                    q_freq[token] +=1
                else:
                    q_freq[token] = 1

            query[qid] = q_freq
        return query

    def read_query_tf(self, query_text, stemming=False):
        #test_query_collection = dict(qid, dict(token, freq))
        #key = query token value = tf

        common_words = open(self.dir_path + '\\common_words.txt')
        stoplist = stopwords.words('english')
        stoplist = stoplist + [w.strip().lower() for w in common_words]
        stoplist = set(stoplist)

        if stemming:
            stemmer = PorterStemmer()
            q_tokens = [stemmer.stem(q) for q in re.split("[^\w]+", query_text) if len(q) > 0 and q.isalpha() and q not in stoplist]
        else:
            q_tokens = [q for q in re.split("[^\w]+", query_text) if len(q) > 0 and q.isalpha() and q not in stoplist]

        #print(q_tokens)
        q_freq = {}
        max = 0
        for token in q_tokens:
            if token in q_freq.keys():
                q_freq[token] += 1
            else:
                q_freq[token] = 1
            if q_freq[token] > max:
                max = q_freq[token]

        q_tf = {}

        for token, freq in q_freq.items():
            q_tf[token] = freq / max

        return q_tf


