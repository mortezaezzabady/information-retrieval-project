from flask import request

from config import CONFIGURATION
from func.Bm25Ranker import Bm25Ranker
from func.TfidfRanker import TfIdfRanker
import re, bisect


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
        result = []
        for doc_id, score in documents:
            doc = parser.docs[parser.index[doc_id]].copy()
            doc['score'] = score
            doc['title'] = doc['title'][0]
            body = doc['body'][0]
            arr = [m.start() for m in re.finditer(query, body)]
            if len(body) > 0 and len(arr) > 0:
                mx = (0, 0)
                ln = 160
                off = 20
                for i in range(len(arr)):
                    j = bisect.bisect_left(arr, arr[i] + ln)
                    if j - i > mx[0]:
                        mx = (j - i, i)
                beg = body[: max(0, arr[mx[1]] - off)].rfind(' ') + 1
                end = body[: min(len(body), arr[mx[1]] + ln + off)].rfind(' ')
                sub = body[beg: end]
                sub = sub.replace(query, '<b>' + query + '</b>')
                doc['body'] = '...' + sub + '...'
            else:
                doc['body'] = body[: 100] + '...'
            result.append(doc)
        print(len(result))
        return result
