from config import CONFIGURATION
import joblib

from func.bk_indexer import Indexer


class LanguageModel(object):

    def __init__(self, N):
        self.N = N - 1
        self.ngram = {}
        self.vocab_size = 0
        self.landa = CONFIGURATION['hyper_parameters']['landa']

    def populate_counts(self):
        for w in self.ngram:
            self.vocab_size += len(self.ngram[w])

    def update(self, content):
        content += ['<END>']
        for i in range(len(content) - self.N):
            seq = ' '.join(content[i: i + self.N])
            if seq in self.ngram:
                try:
                    self.ngram[seq][content[i + self.N]] += 1
                except:
                    self.ngram[seq][content[i + self.N]] = 1
            else:
                self.ngram[seq] = {content[i + self.N]: 1}

    def build(self, docs):
        for document in docs:
            for t in list(CONFIGURATION['weights'].keys()):
                self.update(Indexer.clean(document[t][0]))
        self.populate_counts()

    def C(self, word, context):
        return self.ngram.get(context, {word: 0}).get(word, 0)

    def S(self, words):
        s = sum(self.ngram.get(words, {}).values())
        if words == '':
            s -= 1
        return s

    def P(self, words, uni_lm=None):
        if len(words) == 1 and uni_lm is not None:
            return uni_lm.P(words)
        p = 1.0
        mark = False
        for i in range(len(words) - self.N):
            seq = ' '.join(words[i: i + self.N])
            if uni_lm is None:
                p *= ((self.C(words[i + self.N], seq) / self.S(seq)) if self.S(seq) > 0 else 0)
            else:
                p *= ((self.C(words[i + self.N], seq) / self.S(seq)) if self.S(seq) > 0 else 0) * self.landa + (
                            1 - self.landa) * uni_lm.P(
                    [words[i + self.N]])
            mark = True
        return p if mark else 0.0

    def save(self, filename):
        with open(filename, 'wb') as f:
            joblib.dump(self.ngram, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.ngram = joblib.load(f)
        self.populate_counts()

    @staticmethod
    def from_file(filename, N):
        lm = LanguageModel(N)
        lm.load(filename)
        return lm

    def is_in_language(self, word):
        if self.N == 0:
            return word in list(self.ngram[''].keys())
        return word in list(self.ngram.keys())
