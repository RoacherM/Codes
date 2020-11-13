#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/work_vectors/genvector.py
#    Description:     一个基于gensim库中skip-gram+negetive-sampling算法的词向量训练程序
#    Author:     WY
#    Date: 2020/06/01
#    LastEditTime: 2020/07/01
# -------------------------------------------------

import os
import logging
from gensim.models import Word2Vec
from gensim.models import FastText

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class WordVectors():
    """词向量生成算法
    """
    def __init__(self):
        self.path = '/project/VECT'

    def Word2Vec(self, fpath):
        # -------------------------------------------------
        #    description: Word2Vec生成词向量算法
        #    param fpath 参数保存路径
        #    return:
        # -------------------------------------------------
        print('初始化Word2Vec模型...')
        model = Word2Vec(corpus_file=fpath,
                         size=256,
                         window=5,
                         min_count=25,
                         iter=60,
                         sg=1,
                         hs=1,
                         workers=10)
        print('训练Word2Vec词向量...')
        dirs = os.path.join(self.path, 'sgns')
        if not os.path.exists(dirs):
            os.mkdir(dirs)
        model.save(os.path.join(dirs, 'Word2Vec.model'))
        model.wv.save_word2vec_format(os.path.join(dirs, 'Word2Vec.vector'),
                                      binary=False)
        print('模型参数保存完毕！')

    def FastVec(self, fpath):
        # -------------------------------------------------
        #    description: FastVec生成词向量方法
        #    param fpath 参数保存路径
        #    return:
        # -------------------------------------------------
        print('初始化FastVec模型...')
        model = FastText(corpus_file=fpath,
                         size=256,
                         window=5,
                         min_count=25,
                         iter=60,
                         sg=1,
                         hs=1,
                         workers=10)
        print('训练FastVec词向量...')
        # sentences=word2vec.Text8Corpus(fpath)
        # model.build_vocab(sentences=sentences)
        dirs = os.path.join(self.path, 'fast')
        if not os.path.exists(dirs):
            os.mkdir(dirs)
        model.save(os.path.join(dirs, 'FastVec.model'))
        model.wv.save_word2vec_format(os.path.join(dirs, 'FastVec.vector'),
                                      binary=False)
        print('模型参数保存完毕！')

    def GloVec(self):
        pass


if __name__ == '__main__':

    wv = WordVectors()
    wv.Word2Vec(fpath='/project/CORPUS/corpus.txt')

    # model = Word2Vec.load('./wordvectors/Word2Vec.model')
    # print(model.wv.most_similar(u'银行'))
    # print(model.similarity(u'报警',u'报警人'))
