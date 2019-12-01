import operator

from config import CONFIGURATION
from func.bk_ranker import Ranker
import numpy as np
import joblib


class TfIdfRanker(Ranker):

    @staticmethod
    def get_top_docs(index, parser, tokens, tf_type='r', idf_type='n', dis_func='m', batch_count=1):
        if tf_type == 'r':
            tf_type = 'relative'
        elif tf_type == 'l':
            tf_type = 'logarithm'
        elif tf_type == 'a':
            tf_type = 'augmented'
        else:
            return []

        if dis_func not in ['m', 'c']:
            return []

        if idf_type == 'n':
            idf_type = 'idf'
        elif idf_type == 'p':
            idf_type = 'prob_idf'
        else:
            return []

        documents = {}
        result = []
        if dis_func == 'm':
            for token in tokens:
                relevant_docs = index.get_docs_for_token(token)
                for doc_id, freq in relevant_docs:
                    if doc_id not in documents:
                        documents[doc_id] = 0.
                    tf = index.inverted_index[token]['posting'][doc_id][tf_type]
                    idf = index.inverted_index[token][idf_type]
                    weight = index.inverted_index[token]['posting'][doc_id]['weight']
                    documents[doc_id] += tf * idf * weight
        elif dis_func == 'c':
            doc_set = set()
            query_vec = np.zeros(len(index.vocab))
            for i in range(len(index.vocab)):
                word = index.vocab[i]
                if word in tokens:
                    query_vec[i] = index.inverted_index[word]['idf']
            for token in tokens:
                doc_set |= set([doc_id for doc_id, freq in index.get_docs_for_token(token)])
            flag = [False for i in range(batch_count)]
            bs = len(parser.docs) // batch_count
            for doc_id in doc_set:
                ind = parser.index[doc_id]
                flag[min(ind // bs, batch_count - 1)] = True  # TODO: convert to dict
            for i in range(len(flag)):
                if flag[i]:
                    batch = None
                    with open(CONFIGURATION['path_data'] + '/doc_vec_' + str(i) + '.bin', 'rb') as f:
                        batch = joblib.load(f)
                    st = i * bs
                    for doc_id in doc_set:
                        ind = parser.index[doc_id]
                        if min(ind // bs, batch_count - 1) == i:
                            documents[doc_id] = Ranker.cosine_similarity(batch[ind - st], query_vec)
                    del batch
        for doc_id, score in sorted(documents.items(), key=operator.itemgetter(1), reverse=True):
            doc = parser.docs[parser.index[doc_id]].copy()
            doc['score'] = score
            doc['title'] = doc['title'][0]
            doc['body'] = doc['body'][0]
            result.append(doc)
        print(len(result))
        return result

    @staticmethod
    def vectorization_docs(index, parser, batch_count, tf_type='r'):
        if tf_type == 'r':
            tf_type = 'relative'
        elif tf_type == 'l':
            tf_type = 'logarithm'
        elif tf_type == 'a':
            tf_type = 'augmented'
        else:
            return
        bs = len(parser.docs) // batch_count
        for j in range(batch_count):
            print(j)
            st = j * bs
            en = st + bs
            if j == batch_count - 1:
                en = len(parser.docs)
            doc_vectors = np.zeros((en - st, len(index.vocab)))
            for i in range(len(index.vocab)):
                term = index.vocab[i]
                docs = list(index.inverted_index[term]['posting'].keys())
                idf = index.inverted_index[term]['idf']
                for doc_id in docs:
                    d_ind = parser.index[doc_id]
                    if d_ind < st or d_ind >= en:
                        continue
                    tf = index.inverted_index[term]['posting'][doc_id][tf_type]
                    weight = index.inverted_index[term]['posting'][doc_id]['weight']
                    doc_vectors[d_ind - st][i] = tf * idf * weight
            with open('doc_vec_' + str(j) + '.bin', 'wb') as f:
                joblib.dump(doc_vectors, f)
            del doc_vectors
