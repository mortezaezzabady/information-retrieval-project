import abc
import numpy as np
import math

from func.bk_indexer import Indexer
from func.bk_parser import Parser


class Ranker(abc.ABC):

    def __init__(self):
        pass

    @staticmethod
    @abc.abstractclassmethod
    def get_top_docs(index, parser, tokens, tf_type='r', idf_type='n', dis_func='m', batch_count=1):
        pass

    @staticmethod
    @abc.abstractclassmethod
    def vectorization_docs(index, parser, batch_count, tf_type='r'):
        pass

    @classmethod
    def search(cls, index, parser, query, count=10, tf_type='r', idf_type='n', dis_func='m', batch_count=1):
        assert isinstance(index, Indexer)
        assert isinstance(parser, Parser)
        tokens = Indexer.clean(query)
        top_docs = list(cls.get_top_docs(index, parser, tokens, tf_type, idf_type, dis_func, batch_count))
        return top_docs[:count]

    @staticmethod
    def l2_norm(a):
        return math.sqrt(np.dot(a, a))

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (Ranker.l2_norm(a) * Ranker.l2_norm(b))