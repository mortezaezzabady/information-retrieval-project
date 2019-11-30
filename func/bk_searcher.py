from flask import request

from config import CONFIGURATION
from func.Bm25Ranker import Bm25Ranker
from func.TfidfRanker import TfIdfRanker


class Searcher(object):

    def __init__(self):
        pass

    @staticmethod
    def search(query, indexer, parser):
        tf_type = 'r'
        idf_type = 'n'
        dis_func = 'm'
        if 'tf_type' in request.args and request.args['tf_type'] in ['r', 'a', 'l']:
            tf_type = request.args['tf_type']
        if 'idf_type' in request.args and request.args['idf_type'] in ['n', 'p']:
            idf_type = request.args['idf_type']
        if 'dis_func' in request.args and request.args['dis_func'] in ['m', 'c']:
            dis_func = request.args['dis_func']
        if 'ranker' in request.args and request.args['ranker'] == 'b':
            documents = Bm25Ranker.search(indexer, parser, query, 10, tf_type, idf_type, dis_func,
                                          CONFIGURATION['batches_count'])
        else:
            documents = TfIdfRanker.search(indexer, parser, query, 10, tf_type, idf_type, dis_func,
                                           CONFIGURATION['batches_count'])
        return documents
