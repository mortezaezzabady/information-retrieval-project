import sys
from os import listdir
from os.path import isfile, isdir, join
from time import time

from flask import Flask, request, Response
import json

from config import CONFIGURATION
from func.bk_indexer import Indexer
from func.bk_language_model import LanguageModel
from func.bk_parser import Parser
from func.bk_query_checker import QueryChecker
from func.bk_searcher import Searcher

app = Flask(__name__)
indexer = Indexer()
parser = Parser()
query_checker = None


def is_data_file(path):
    return isfile(path) and path.split('\\')[-1].startswith('WebIR') and path.split('\\')[-1].endswith('.xml')


def main():  # TODO: verbose
    global parser, indexer, query_checker

    if isfile(CONFIGURATION['file_parse']):
        st = time()
        parser = Parser.from_file(CONFIGURATION['file_parse'])
        print('load parse:', time() - st)
    else:
        path = CONFIGURATION['path_documents']
        st = time()
        if isdir(path):
            files = [join(path, f) for f in listdir(path) if is_data_file(join(path, f))]
            for f in files:
                parser.parse(f)
        parser.save(CONFIGURATION['file_parse'])
        print('parse time:', time() - st)

    if isfile(CONFIGURATION['file_index']):
        st = time()
        indexer = Indexer.from_file(CONFIGURATION['file_index'])
        print('load index:', time() - st)
    else:
        st = time()
        indexer.build(parser.docs)
        indexer.calc()
        print('full inverted index size:', len(indexer.inverted_index))
        indexer.remove_minors(10)
        print('index time:', time() - st)
        print(len(indexer.inverted_index), len(indexer.vocab))
        indexer.save(CONFIGURATION['file_index'])
        print('store index time:', time() - st)

    if not isfile(CONFIGURATION['file_unigram_lm']):
        st = time()
        uni_lm = LanguageModel(1)
        uni_lm.build(parser.docs)
        uni_lm.save(CONFIGURATION['file_unigram_lm'])
        print(uni_lm.vocab_size)
        print('unigram language model time:', time() - st)

    if not isfile(CONFIGURATION['file_bigram_lm']):
        st = time()
        bi_lm = LanguageModel(2)
        bi_lm.build(parser.docs)
        bi_lm.save(CONFIGURATION['file_bigram_lm'])
        print(bi_lm.vocab_size)
        print('bigram language model time:', time() - st)

    query_checker = QueryChecker(indexer)


@app.route('/api/v1/search', methods=['GET'])
def search():
    if 'query' in request.args and len(request.args['query']) > 0:
        queries = [request.args['query']]
        suggestion = ''
        active = False
        if 'force' not in request.args or request.args['force'] == 0:
            queries, suggestion, active = query_checker.check(request.args['query'], indexer)
        results = []
        correction = None
        if active or len(suggestion) > 0:
            correction = {
                'active': active,
                'query':  suggestion
            }
        for query in queries:
            results += Searcher.search(query, indexer, parser)
        resp = Response(json.dumps({
            'success': True,
            'result': {
                'documents': results,
                'correction': correction,
            }
        }))
    else:
        resp = Response(json.dumps({
            'success': False,
            'error': 'BAD_REQUEST'
        }))
    return resp


@app.route('/api/v1/suggestion', methods=['GET'])
def suggest():
    if 'query' in request.args and len(request.args['query']) > 0:
        resp = Response(json.dumps({
            'success': True,
            'suggestions': query_checker.suggest(request.args['query'])
        }))
    else:
        resp = Response(json.dumps({
            'success': False,
            'error': 'BAD_REQUEST'
        }))
    return resp


@app.route('/')
def hello_world():
    print(len(indexer.vocab))
    return 'Hello World!'


@app.after_request
def apply_caching(response):
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, x-access-token'
    return response


if __name__ == '__main__':
    main()
    app.run('0.0.0.0', '5000')
