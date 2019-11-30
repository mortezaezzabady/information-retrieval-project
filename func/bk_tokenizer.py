from hazm import word_tokenize, Normalizer
import re

normalizer = Normalizer()


class Tokenizer(object):
    nums = {'۰': 'صفر', '۱': 'یک', '۲': 'دو', '۳': 'سه', '۴': 'چهار', '۵': 'پنج', '۶': 'شش', '۷': 'هفت', '۸': 'هشت',
            '۹': 'نه'}

    def __init__(self):
        pass

    def tokenize(self, text):
        text = self.remove_symbols(text)
        text = re.sub('\s+', ' ', text).strip()
        text = text.lower()
        text = text.replace('\u200c', ' ').replace('\n', '').replace('\r', '').replace('ي', 'ی').replace('ك', 'ک')
        normalized_text = normalizer.normalize(text)
        return word_tokenize(normalized_text)

    def remove_symbols(self, text):
        text = text.replace('.', ' ').replace('؟', ' ').replace('?', ' ').replace(')', ' ').replace('(', ' ').replace(
            '»', ' ')
        text = text.replace('«', ' ').replace('<', ' ').replace('>', ' ').replace('،', ' ').replace('-', ' ').replace(
            '|', ' ')
        text = text.replace('[', ' ').replace(']', ' ').replace('{', ' ').replace('}', ' ').replace(',', ' ').replace(
            '/', ' ')
        text = text.replace('؛', ' ').replace('+', ' ').replace('!', ' ').replace('ء', ' ').replace('_', ' ').replace(
            ';', ' ')
        text = text.replace('\u200f', ' ').replace('\u200d', ' ').replace('=', ' ').replace(':', ' ') \
            .replace('–', ' ').replace('*', ' ')
        text = text.replace('«', ' ').replace('»', ' ').replace('\'', ' ')
        return text

    def num_to_word(self, tokens):
        nums = {'۰': 'صفر', '۱': 'یک', '۲': 'دو', '۳': 'سه', '۴': 'چهار', '۵': 'پنج', '۶': 'شش', '۷': 'هفت', '۸': 'هشت',
                '۹': 'نه'}
        for num, word in nums.items():
            for i in range(len(tokens)):
                token = tokens[i]
                if token in list(nums.keys()):
                    tokens[i] = nums[token]
        return tokens
