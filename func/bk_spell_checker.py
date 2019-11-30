

class SpellChecker(object):
    langs = ['en', 'fa']
    qwerty_keyboard = {'en': [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
                              ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
                              ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',']],
                       'fa': [['ض', 'ص', 'ث', 'ق', 'ف', 'غ', 'ع', 'ه', 'خ', 'ح', 'ج', 'چ', 'پ'],
                              ['ش', 'س', 'ی', 'ب', 'ل', 'ا', 'ت', 'ن', 'م', 'ک', 'گ'],
                              ['ظ', 'ط', 'ز', 'ر', 'ذ', 'د', 'ئ', 'و']]}
    phonetic_clusters = [['ز', 'ذ', 'ظ', 'ض'],
                         ['ص', 'س', 'ث'],
                         ['ح', 'ه'],
                         ['ی', 'ئ'],
                         ['ق', 'غ'],
                         ['ت', 'ط']]
    alphabets = {'fa': ['ض', 'ص', 'ث', 'ق', 'ف', 'غ', 'ع', 'ه', 'خ', 'ح', 'ج', 'چ', 'پ',
                        'ش', 'س', 'ی', 'ب', 'ل', 'ا', 'ت', 'ن', 'م', 'ک', 'گ',
                        'ظ', 'ط', 'ز', 'ر', 'ذ', 'د', 'ئ', 'و', 'ژ', 'آ'],
                 'en': ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                        'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                        'z', 'x', 'c', 'v', 'b', 'n', 'm']}

    def __init__(self):
        pass

    @staticmethod
    def levenshtein_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    @staticmethod
    def dl_distance(s1, s2):
        n = len(s1)
        m = len(s2)
        dp = {}
        for i in range(-1, n + 1):
            dp[(i, -1)] = i + 1
        for j in range(-1, m + 1):
            dp[(-1, j)] = j + 1

        for i in range(n):
            for j in range(m):
                if s1[i] == s2[j]:
                    cost = 0
                else:
                    cost = 2
                    if s2[j] in SpellChecker.adjacents(s1[i])[0]:
                        cost = 1
                dp[(i, j)] = min(
                    dp[(i - 1, j)] + 1,  # deletion
                    dp[(i, j - 1)] + 1,  # insertion
                    dp[(i - 1, j - 1)] + cost)  # substitution
                if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                    dp[(i, j)] = min(dp[(i, j)], dp[i - 2, j - 2] + 1)  # transposition

        return dp[n - 1, m - 1]

    @staticmethod
    def adjacents(c):
        adjs = []
        for cluster in SpellChecker.phonetic_clusters:
            if c in cluster:
                adjs = [x for x in cluster if x != c]
        if c == 'آ':
            c = 'ا'
        if c == 'ژ':
            c = 'ز'
        steps = [[(0, -1), (0, 1), (1, 0), (1, -1)],
                 [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)],
                 [(0, -1), (0, 1), (-1, 0), (-1, 1)]]
        L = None
        X = -1
        Y = -1
        for lang in SpellChecker.langs:
            for i in range(len(SpellChecker.qwerty_keyboard[lang])):
                try:
                    j = SpellChecker.qwerty_keyboard[lang][i].index(c)
                    L = lang
                    X = i
                    Y = j
                    for x, y in steps[i]:
                        if j + y >= 0 and j + y < len(SpellChecker.qwerty_keyboard[lang][i]):
                            adjs.append(SpellChecker.qwerty_keyboard[lang][i + x][j + y])
                except:
                    continue
        other_langs = {}
        for lang in SpellChecker.langs:
            if lang != L:
                other_langs[lang] = SpellChecker.qwerty_keyboard[lang][X][Y]
        return adjs, other_langs

    @staticmethod
    def edits1(word):
        lang = SpellChecker.detect_language(word)
        letters = SpellChecker.alphabets[lang]
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in SpellChecker.adjacents(R[0])[0]]
        inserts = [L + c + R for L, R in splits for c in letters]
        others = []
        for l in SpellChecker.langs:
            if l != lang:
                s = ''
                mark = False
                for c in word:
                    adjacents = SpellChecker.adjacents(c)[1]
                    if l in list(adjacents.keys()):
                        s += adjacents[l]
                    else:
                        mark = True
                if mark:
                    continue
                others.append(s)
        return set(deletes + transposes + replaces + inserts), set(others)

    @staticmethod
    def edits2(word):
        return set(e2 for e1 in SpellChecker.edits1(word)[0] for e2 in SpellChecker.edits1(e1)[0])

    @staticmethod
    def detect_language(word):
        for lang in SpellChecker.langs:
            for i in range(len(SpellChecker.qwerty_keyboard[lang])):
                try:
                    j = SpellChecker.qwerty_keyboard[lang][i].index(word[0])
                    return lang
                except:
                    continue
        return 'fa'
