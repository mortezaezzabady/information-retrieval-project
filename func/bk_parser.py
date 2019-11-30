import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import pickle

from func.bk_indexer import Indexer


class Parser(object):

    def __init__(self):
        self.docs = []
        self.index = {}
        self.avdl = 0

    def doc_count(self):
        return len(self.docs)

    def parse(self, filename):
        root = ET.parse(filename).getroot()
        for doc in root.findall('DOC'):
            docId = doc.find('DOCID').text
            url = doc.find('URL').text
            html = doc.find('HTML').text
            html = re.sub('(<!--.*?-->)', '', html, flags=re.DOTALL)
            soup = BeautifulSoup(html, 'html.parser')
            for s in soup(['script', 'style']):
                s.decompose()
            title = ''
            body = ''
            if soup.find('title') is not None:
                title = soup.title.text
            if soup.find('body') is not None:
                body = soup.body.text
            self.docs.append({'id': docId, 'url': url, 'title': (title, len(Indexer.clean(title))),
                              'body': (body, len(Indexer.clean(body)))})

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.docs, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.docs = pickle.load(f)
            self.avdl = 0.
            for i in range(len(self.docs)):
                self.index[self.docs[i]['id']] = i
                self.avdl += self.docs[i]['title'][1] + self.docs[i]['body'][1]
            self.avdl /= len(self.docs)

    @staticmethod
    def from_file(filename):
        parser = Parser()
        parser.load(filename)
        return parser
