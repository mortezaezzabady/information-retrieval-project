import operator
from func.bk_ranker import Ranker
from config import CONFIGURATION


class Bm25Ranker(Ranker):

    @staticmethod
    def get_top_docs(index, parser, tokens, tf_type='r', idf_type='n', dis_func='m', batch_count=1):
        k1 = CONFIGURATION['hyper_parameters']['k1']
        b = CONFIGURATION['hyper_parameters']['b']
        avdl = parser.avdl
        documents = {}
        for token in tokens:
            relevant_docs = index.get_docs_for_token(token)
            for doc_id, freq in relevant_docs:
                if doc_id not in documents:
                    documents[doc_id] = 0.
                doc = parser.docs[parser.index[doc_id]]
                tf = index.inverted_index[token]['posting'][doc_id]['tf']
                idf = index.inverted_index[token]['prob_idf']
                weight = index.inverted_index[token]['posting'][doc_id]['weight']
                doc_size = doc['title'][1] + doc['body'][1]
                documents[doc_id] += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_size / avdl)))) * weight
        return sorted(documents.items(), key=operator.itemgetter(1), reverse=True)
