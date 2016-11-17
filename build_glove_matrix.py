import os
import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
GLOVE_DIR = os.path.join(PKG, 'glove/')
DIC_FILE = os.path.join(PKG, 'splitdata/All.dic.body')

class BuildEmbedMatrix(object):
    def __init__(self): 
        self.embedding_index = {}
        self.words_index = {}
        self.embed_dim = 0
        self.embedding_matrix = np.zeros(1)
        self.available_datasets = ['stack','interspeech']

    def _readGlove(self, glove_filename):
        f = open(os.path.join(GLOVE_DIR, glove_filename), 'r')
        first_line = f.readline().split()
        print first_line
        self.embed_dim = int(first_line[1])
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embedding_index[word] = coefs
        f.close()
        print 'Found %s word vectors.' % len(self.embedding_index)

    def _readWordDic(self):
        f = open(DIC_FILE, 'r')
        """0 is reserved for words NOT in dic!!!"""
        counter = 1
        for line in f.readlines():
            word = line.split()[0]
            self.words_index.update({word:counter})
            counter += 1
        f.close()

    def buildEmbedMatrix(self, glove_filename):
        print 'Using %s to build mat' % glove_filename
        self._readGlove(glove_filename)
        self._readWordDic()
        self.embedding_matrix = np.zeros((len(self.words_index) + 1, self.embed_dim), dtype='float32')
        for word, i in self.words_index.items():
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embeddings will be all zeros
                self.embedding_matrix[i] = embedding_vector

        return self.embedding_matrix

if __name__ == '__main__':
    print 'Demo of BuildEmbedMatrix'
    BEM = BuildEmbedMatrix()
    embedding_matrix = BEM.buildEmbedMatrix('glove.6B.100d.txt')
