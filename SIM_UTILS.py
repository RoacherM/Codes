#!/opt/miniconda/bin/python python
# coding: utf-8

# -------------------------------------------------
#    File Name：     SIM_UTILS.py
#    Description :   语义计算引擎，目前采用两种方式，基于simBERT预训练通用模型和基于警情语料的simVECT模型
#    Author :        WY
#    date：          2020/6/13
# -------------------------------------------------

import json
import codecs
import numpy as np
from CUT_UTILS import cut

# simVECT的模型参数
embeddings = '/Users/wonbyron/Desktop/work/Codes/project/VECT/tmp/embeddings.npy'
id2word = '/Users/wonbyron/Desktop/work/Codes/project/VECT/tmp/id2word.json'
word2id = '/Users/wonbyron/Desktop/work/Codes/project/VECT/tmp/word2id.json'
# simBERT的模型参数
bert_config_path = '/Users/wonbyron/bert/chinese_simbert_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = '/Users/wonbyron/bert/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = '/Users/wonbyron/bert/chinese_simbert_L-12_H-768_A-12/vocab.txt'


def normalize(x):
    '''x为一个/组向量，返回归一化后的结果'''
    if len(x.shape) > 1:
        return x / np.clip(
            x**2, 1e-12, None).sum(axis=1).reshape((-1, 1) + x.shape[2:])**0.5
    else:
        return x / np.clip(x**2, 1e-12, None).sum()**0.5


def topK(arr, n):
    '''求数组的前top个值及其索引
    # x = np.array([1,0,3,9])
    # xs = np.sin(np.arange(9)).reshape((3, 3))
    # print(xs)
    # print(topK(x,3))
    # print(topK(xs,3))

    # print(xs[topK(xs,3)])
    # print(x[topK(x,3)[0]])
    '''
    # 解索引
    flat = arr.flatten()
    # 求前k个最大值的索引
    indices = np.argpartition(flat, -n)[-n:]
    # 索引排序
    indices = indices[np.argsort(-flat[indices])]
    # 求每个索引在原数组中的对应位置
    return np.unravel_index(indices, arr.shape)


def matrixD(mat_a, mat_b, similarity=True):
    # -------------------------------------------------
    #    description: 快速计算矩阵行与行之间的距离及相似度算法
    #    param mat_a array A*N的矩阵
    #    param mat_b array B*N的矩阵
    #    param similarity boolean similairty为True时返回余弦相似度，否则为欧式距离
    #    return: A*B的矩阵，表示A的每一行和B的每一行之间的距离或相似度
    # -------------------------------------------------
    la = mat_a.shape[0]
    lb = mat_b.shape[0]
    dists = np.zeros((la, lb))
    dists = np.sqrt(-2 * np.dot(mat_a, mat_b.T) +
                    np.sum(np.square(mat_b), axis=1) +
                    np.transpose([np.sum(np.square(mat_a), axis=1)]))
    if similarity:
        dists = 1 - dists * dists / 2
        return dists
    return dists


class simVECT(object):
    '''基于词向量的语义搜索引擎'''
    def __init__(self,
                 embeddings=embeddings,
                 id2word=id2word,
                 word2id=word2id):

        self.embeddings = np.load(embeddings)
        self.word_size = len(self.embeddings)
        with codecs.open(id2word, 'r', encoding='utf-8') as fp:
            self.id2word = json.load(fp)
        with codecs.open(word2id, 'r', encoding='utf-8') as fp:
            self.word2id = json.load(fp)
        self.nb_context_words = None

    def most_correlative(self, word, topn=10, with_sim=True):
        # -------------------------------------------------
        #    description: 计算相关词语
        #    param word str，待计算的词语
        #    param topn int，返回前topn个词语
        #    param with_sim boolean，当为True时，返回相似度
        #    return:
        # -------------------------------------------------
        word_vec = self.embeddings[self.word2id[word]]
        word_sim = np.dot(self.embeddings, word_vec)
        word_sim_sort = word_sim.argsort()[::-1]
        if with_sim:
            return [(self.id2word[str(i)], word_sim[i])
                    for i in word_sim_sort[:topn]]
        return [self.id2word[str(i)] for i in word_sim_sort[:topn]]

    def most_similar(self,
                     word,
                     topn=10,
                     nb_context_words=100000,
                     with_sim=True):
        # -------------------------------------------------
        #    description: 计算相似词语
        #    param word str，输入词语
        #    param topn int，返回前topn个词语
        #    param nb_context_words int，默认为100000，限制词语范围
        #    return:
        # -------------------------------------------------
        if nb_context_words != self.nb_context_words:
            embeddings_ = self.embeddings[:nb_context_words]
            embeddings_ = embeddings_ - embeddings_.mean(axis=0)
            U = np.dot(embeddings_.T, embeddings_)
            U = np.linalg.cholesky(U)
            embeds = np.dot(self.embeddings, U)
            self.nb_context_words = nb_context_words
            self.normalized_embeddings = embeds / (embeds**2).sum(axis=1).reshape((-1, 1))**0.5
        word_vec = self.normalized_embeddings[self.word2id[word]]
        word_sim = np.dot(self.normalized_embeddings, word_vec)
        word_sim_sort = word_sim.argsort()[::-1]
        if with_sim:
            return [(self.id2word[str(i)], word_sim[i])
                    for i in word_sim_sort[:topn]]
        return [self.id2word[str(i)] for i in word_sim_sort[:topn]]

    def analogy(self,
                pos_word_1,
                pos_word_2,
                neg_word=None,
                topn=10,
                with_sim=True):
        # -------------------------------------------------
        #    description: 线性加减计算，实现类似'国王'+'女人'-'皇冠'='女王'的效果，目前效果一般
        #    param pos_word_1 str，词语，如国王
        #    param pos_word_2 str，词语，如女人
        #    param neg_word str，词语，如皇冠
        #    return:
        # -------------------------------------------------
        if neg_word:
            word_vec = self.embeddings[
                self.word2id[pos_word_1]] + self.embeddings[self.word2id[
                    pos_word_2]] - self.embeddings[self.word2id[neg_word]]
        else:
            word_vec = self.embeddings[self.word2id[
                pos_word_1]] + self.embeddings[self.word2id[pos_word_2]]
        word_vec = word_vec / np.dot(word_vec, word_vec)**0.5
        word_sim = np.dot(self.embeddings, word_vec)
        word_sim_sort = word_sim.argsort()[::-1]
        if with_sim:
            return [(self.id2word[str(i)], word_sim[i])
                    for i in word_sim_sort[:topn]]
        return [self.id2word[str(i)] for i in word_sim_sort[:topn]]

    def sent2vec(self, sent):
        # -------------------------------------------------
        #    description: 句子转向量
        #    param sent str/list 一个/一组句子
        #    return:
        # -------------------------------------------------
        Z = []
        if isinstance(sent, list):
            for s in sent:
                s = cut(s)
                idxs = [self.word2id[w] for w in s if w in self.word2id]
                sv = self.embeddings[idxs].sum(axis=0)
                Z.append(sv)
        else:
            sent = cut(sent)
            idxs = [self.word2id[w] for w in sent if w in self.word2id]
            sv = self.embeddings[idxs].sum(axis=0)
            Z.append(sv)
        return normalize(np.array(Z))

    def keywords(self, token=None, text='', topn=1, with_sim=True):
        # -------------------------------------------------
        #    description: 关键词匹配算法，当有token时，返回token中和句子相似度最大的词语；当无token时，返回句中关键词
        #    param token list，待选token，默认为空，token中的词语需要包含在词向量词表中
        #    param text str，输入文本
        #    param topn int，默认为1，只返回匹配度最大的词语，不可超过token最大长度范围
        #    param with_sim boolean，返回带相似度的结果
        #    return:
        # -------------------------------------------------
        if token is not None:
            r = token
        else:
            r = token = [
                c for c in cut(text) if len(c) > 1 and c in self.word2id
            ]
        X = []
        for t in r:
            X.append(self.word2id[t])
        # print(X)
        # 获得句子的词向量
        Z = self.embeddings[X]
        score = np.dot(self.sent2vec(text), Z.T)
        # print(score.shape)
        if with_sim:
            return [(token[i], score[0][i]) for i in topK(score, topn)[1]]
        return np.array(token)[topK(score, topn)[1]]

    def sentence_similarity(self, sent_1, sent_2):
        # -------------------------------------------------
        #    description: 句子相似度匹配
        #    param sent_1 str，输入句子
        #    param sent_2 str，输入句子
        #    return:
        # -------------------------------------------------
        sent_vec_1 = self.sent2vec(sent_1)
        sent_vec_2 = self.sent2vec(sent_2)
        return np.dot(sent_vec_1, sent_vec_2.T)


class simBERT(object):
    '''基于BERT的语义计算引擎'''
    def __init__(self,
                 config=bert_config_path,
                 checkpoint=bert_checkpoint_path,
                 dicts=bert_dict_path):

        from bert4keras.backend import keras
        from bert4keras.tokenizers import Tokenizer
        from bert4keras.snippets import sequence_padding
        from bert4keras.models import build_transformer_model

        self.config_path = config
        self.checkpoint_path = checkpoint
        self.dict_path = dicts
        self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
        self.sequence_padding = sequence_padding
        self.bert = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            with_pool='linear',
            application='unlim',
            return_keras_model=False,
        )
        self.encoder = keras.models.Model(self.bert.model.inputs,
                                          self.bert.model.outputs[0])
        # self.seq2seq = keras.models.Model(self.bert.model.inputs,self.bert.model.outputs[1])

    def sent2vec(self, sent):
        # -------------------------------------------------
        #    description: 句子转向量
        #    param sent str，输入句子
        #    return:
        # -------------------------------------------------
        if isinstance(sent, list):
            X, S = [], []
            for s in sent:
                x, s = self.tokenizer.encode(s)
                X.append(x)
                S.append(s)
            X = self.sequence_padding(X)
            S = self.sequence_padding(S)
            # Z = self.encoder.predict([X,S])
        else:
            x, s = self.tokenizer.encode(sent)
            X = self.sequence_padding([x])
            S = self.sequence_padding([s])
        Z = self.encoder.predict([X, S], verbose=1)
        # 将向量归一化，便于计算各类距离
        return normalize(Z)

    def keywords(self, token=None, text='', topn=1, with_sim=True):
        # -------------------------------------------------
        #    description: 关键词匹配算法，当有token时，返回token中和句子相似度最大的词语；当无token时，返回句中关键词
        #    param token list，待选词表，可为空
        #    param text str，输入文本
        #    param topn int，默认为1，最大不超过token词表的最大长度
        #    param with_sim boolean 当为True时，返回带相似度的结果
        #    return:
        # -------------------------------------------------
        if token is not None:
            r = token + [text]
            # r = token + [c for c in cut(text) if len(c) > 1]
        else:
            token = [c for c in cut(text) if len(c) > 1]
            r = token + token
        X, S = [], []
        for t in r:
            x, s = self.tokenizer.encode(t)
            X.append(x)
            S.append(s)
        X = self.sequence_padding(X)
        S = self.sequence_padding(S)
        Z = normalize(self.encoder.predict([X, S]))
        score = np.dot(Z[len(token):], Z[:len(token)].T)
        # print(score.shape)
        if with_sim:
            return [(token[i], score[0][i]) for i in topK(score, topn)[1]]
        return np.array(token)[topK(score, topn)[1]]

    def sentence_similarity(self, sent_1, sent_2):
        # -------------------------------------------------
        #    description: 句子相似度计算
        #    param sent_1 str，输入语句
        #    param sent_2 str，输入语句
        #    return:
        # -------------------------------------------------
        sent_vec_1 = self.sent2vec(sent_1)
        sent_vec_2 = self.sent2vec(sent_2)
        similarity = np.dot(sent_vec_1, sent_vec_2.T)
        return similarity[0][0]


if __name__ == '__main__':

    bt = simBERT()
    print(
        bt.sentence_similarity('噪声扰民',
                               '沅江路菜场门口,一乞讨人员在用高音喇叭播放音乐，扰民（接警台电话：25609）'))
    print(
        bt.keywords(token=['卖艺', '噪声扰民', '乞讨', '盗窃'],
                    text='沅江路菜场门口,一乞讨人员在用高音喇叭播放音乐，扰民（接警台电话：25609）',
                    topn=3))
    print(
        bt.keywords(token=None,
                    text='沅江路菜场门口,一乞讨人员在用高音喇叭播放音乐，扰民（接警台电话：25609）',
                    topn=3))
    print(bt.sentence_similarity('流浪', '流浪'))
    wv = simVECT()
    print(wv.most_similar('东海县', with_sim=False))
    print(wv.most_correlative('盗窃', with_sim=False))
    print(
        wv.keywords(token=['扰民', '乞讨', '事故'],
                    text='沅江路菜场门口,一乞讨人员在用高音喇叭播放音乐，扰民（接警台电话：25609）',
                    topn=3))
    print(wv.sentence_similarity('流浪', '2018年03月02日05时30分，民警王益根在巡逻中发现一名流浪人员。'))
    print(wv.analogy(pos_word_1='车辆', pos_word_2='交通', neg_word='非机动车'))