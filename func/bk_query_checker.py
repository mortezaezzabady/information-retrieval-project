from config import CONFIGURATION
from time import time
import itertools

from func.bk_indexer import Indexer
from func.bk_language_model import LanguageModel
from func.bk_spell_checker import SpellChecker


class QueryChecker(object):

    def __init__(self, indexer):
        self.indexer = indexer
        st = time()
        self.uni_lm = LanguageModel.from_file(CONFIGURATION['file_unigram_lm'], 1)
        print('words count:', self.uni_lm.vocab_size)
        print('load unigram language model:', time() - st)

        st = time()
        self.bi_lm = LanguageModel.from_file(CONFIGURATION['file_bigram_lm'], 2)
        print('biwords count:', self.bi_lm.vocab_size)
        print('load bigram language model:', time() - st)

    def correct(self, query):
        st = time()
        clean_query = Indexer.clean(query)
        query_candidates = []
        for word in clean_query:
            e1 = SpellChecker.edits1(word)
            e2 = SpellChecker.edits2(word)
            # print(len(e1[0] | e1[1]), len(e2))
            s = set()
            s = (e1[0] | e1[1] | e2 | {word}) & set(self.indexer.vocab)
            s = list(s)
            # print(len(s))
            cands = []
            for i in range(len(s)):
                cands.append((s[i], self.uni_lm.P([s[i]])))
            cands = sorted(cands, key=lambda k: k[1], reverse=True)[:3]
            # print(cands)
            # wp = uni_lm.P([word]) # TODO: check word prob
            c = [cand[0] for cand in cands]
            query_candidates.append(c)
        candidates_prob = [(query, self.bi_lm.P(clean_query, self.uni_lm))]
        for element in itertools.product(*query_candidates):
            candidates_prob.append((' '.join(element), self.bi_lm.P(element, self.uni_lm)))
        candidates_prob = sorted(candidates_prob, key=lambda k: k[1], reverse=True)
        print('spelling correction:', time() - st)
        return candidates_prob[0]

    def check(self, query, indexer):
        correction = self.correct(query)
        search_query = []
        suggestion = ''
        active = False
        if query == correction[0]:
            search_query = [query]
        elif len(indexer.get_docs_for_query(query)) > CONFIGURATION['docs_count_lim']:
            search_query = [query]
            if correction[1] > CONFIGURATION['prob_lim']:
                suggestion = correction[0]
        elif len(indexer.get_docs_for_query(correction[0])) > CONFIGURATION['docs_count_lim']:
            search_query = [correction[0]]
            active = True
        else:
            search_query = [query, correction[0]]
        return search_query, suggestion, active
