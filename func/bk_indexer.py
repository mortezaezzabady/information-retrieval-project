from hazm import Stemmer, Lemmatizer
import threading
import queue
import math
from config import CONFIGURATION, STOP_WORDS
import joblib

from func.bk_tokenizer import Tokenizer


class Indexer(object):
    tokenizer = Tokenizer()
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()

    def __init__(self, is_uni=True, is_positional=False):
        self.inverted_index = {}
        self.lock = threading.Lock()
        self.docs = set()
        self.term_count = 0
        self.is_uni = is_uni
        self.is_positional = is_positional
        self.index_queue = queue.Queue()
        self.vocab = []
        self.minors = {}

    def doc_count(self):
        return len(self.docs)

    def save(self, filename):
        with open(filename, 'wb') as f:
            joblib.dump(self.inverted_index, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.inverted_index = joblib.load(f)
        if self.is_uni:
            self.load_minors(CONFIGURATION['file_drop'])
        self.vocab = list(self.inverted_index.keys())
        return self.populate_documents()

    def load_minors(self, filename):
        with open(filename, 'rb') as f:
            self.minors = joblib.load(f)

    @staticmethod
    def clean(content, is_uni=True):
        tokens = Indexer.tokenizer.tokenize(content)
        tokens = [Indexer.lemmatizer.lemmatize(i) for i in tokens if
                  i not in STOP_WORDS]  # TODO: don't remove stop words in positional
        tokens = [Indexer.tokenizer.nums[token] if token in list(Indexer.tokenizer.nums.keys()) else token for token in
                  tokens if token not in STOP_WORDS and len(token) > 1]
        if is_uni:
            return tokens
        bi_gram = []
        for i in range(len(tokens) - 1):
            token = tokens[i] + ' ' + tokens[i + 1]
            token = token.replace('\u200c', ' ')
            bi_gram.append(token)
        return bi_gram  # TODO: extended bi gram

    def repopulate_counts(self):
        self.term_count = sum([i[1]['count'] for i in self.inverted_index.items()])

    def populate_documents(self):
        self.docs = set()
        for token in self.inverted_index:
            posting = list(self.inverted_index[token]['posting'].keys())
            for doc in posting:
                self.docs.add(doc)
        self.repopulate_counts()
        return self.doc_count()

    @staticmethod
    def from_file(filename, is_uni=True):
        indexer = Indexer()
        indexer.is_uni = is_uni
        indexer.load(filename)
        return indexer

    def index(self, document):
        tokens = {}
        comp_tokens = []
        for t in list(CONFIGURATION['weights'].keys()):
            tokens[t] = Indexer.clean(document[t][0], self.is_uni)
            comp_tokens += tokens[t]
        token_set = set(comp_tokens)
        if self.is_positional:
            pass  # TODO: positional
        else:
            if self.is_uni:
                max_tf = 0
                for token in token_set:
                    max_tf = max(comp_tokens.count(token), max_tf)
                for token in token_set:
                    count = comp_tokens.count(token)
                    weight = 0
                    for t in list(CONFIGURATION['weights'].keys()):
                        if token in tokens[t]:
                            weight += CONFIGURATION['weights'][t]
                    self.update(token, document, weight, count, max_tf)
            else:
                for token in token_set:
                    count = comp_tokens.count(token)
                    self.update(token, document, 0, count, 0)
        # self.repopulate_counts()
        self.docs.add(document['id'])
        # return self.term_count

    def update(self, token, document, weight, count, max_tf, positions=[]):
        if count == 0:
            return
        if token not in self.inverted_index:
            self.inverted_index[token] = {
                'idf': 0,
                'count': 0,
                'posting': {},
            }
        with self.lock:
            if self.is_positional:
                pass
            else:
                if self.is_uni:
                    tf = count
                    doc_size = 0
                    for t in list(CONFIGURATION['weights'].keys()):
                        doc_size += document[t][1]
                    self.inverted_index[token]['posting'][document['id']] = {
                        'tf': tf,
                        'weight': weight,
                        'relative': 1.0 * tf / doc_size,
                        'augmented': 0.5 + (0.5 * tf / max_tf),
                        'logarithm': 1 + math.log(tf)
                    }
                else:
                    self.inverted_index[token]['posting'][document['id']] = count
            self.inverted_index[token]['idf'] += 1
            self.inverted_index[token]['count'] += count

    def get_docs_for_token(self, token, count=None):
        token = Indexer.clean(token)
        if len(token) == 0:
            return []
        token = token[0]
        if token not in self.inverted_index:
            return []
        d = {}
        docs = self.inverted_index.get(token)['posting']
        sorted_docs = []
        doc_list = []
        if self.is_positional:
            pass
        else:
            sorted_docs = sorted(docs, key=lambda k: docs[k]['tf'], reverse=True)
            doc_list = [(x, docs[x]['tf']) for x in sorted_docs]
        return doc_list if count is None else doc_list[:count]

    def build(self, docs, thread_cnt=8):
        for d in docs:
            self.index_queue.put(d)
        threads = []
        for i in range(thread_cnt):
            th = threading.Thread(target=self.index_worker)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

    def index_worker(self):
        while True:
            try:
                document = self.index_queue.get(timeout=0.1)
            except:
                return
            self.index(document)

    def calc(self):
        for token in self.inverted_index.keys():
            cnt = self.inverted_index[token]['idf']
            if cnt == 0:
                continue
            self.inverted_index[token]['idf'] = math.log((1 + self.doc_count()) / cnt)
            if self.is_uni:
                self.inverted_index[token]['prob_idf'] = math.log((1.0 * self.doc_count() - cnt) / cnt)

    def remove_minors(self, drop=1):
        minors = {}
        new_ii = {}
        for token in self.inverted_index:
            if self.inverted_index[token]['count'] <= drop:
                if self.is_uni:
                    minors[token] = self.inverted_index[token]
            else:
                new_ii[token] = self.inverted_index[token]
        self.inverted_index = new_ii
        if self.is_uni:
            with open(CONFIGURATION['file_drop'], 'wb') as f:
                joblib.dump(minors, f)

    def get_docs_for_query(self, query):
        tokens = Indexer.clean(query)
        doc_set = set()
        for token in tokens:
            if token in self.inverted_index:
                doc_set |= set(self.inverted_index[token]['posting'].keys())
        return doc_set
