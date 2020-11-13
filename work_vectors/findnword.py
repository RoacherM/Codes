#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/work_vectors/datamine.py
#    Description:     基于最大熵模型算法的词汇挖掘程序，支持挖掘指定长度的词汇和指定长度组合的词组
#    Author:     WY
#    Date: 2020/05/30
#    LastEditTime: 2020/07/01
# -------------------------------------------------

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain

# 导入预处理模块
import sys
sys.path.append('.')
# from CUT_UTILS import cut


class Extraction():
    """
    """
    def __init__(self, min_count=100, min_support=50, min_s=2.0, max_sep=3):
        # 录取词语最小出现次数
        self.min_count = min_count
        # 录取词语最低支持度，1代表着随机组合
        self.min_support = min_support
        # 录取词语最低信息熵，越大说明越有可能独立成词
        self.min_s = min_s
        # 候选词语的最大字数,最大支持7个字符查找
        self.max_sep = max_sep

    def words(self, contents):
        # -------------------------------------------------
        #    description: 词汇挖掘算法
        #    param contents str，文本
        #    return:
        # -------------------------------------------------
        rs = []  # 存放最终结果
        rt = []  # 存放临时结果
        # 统计每个词出现的频率
        rs.append(pd.Series(list(contents)).value_counts())
        # 统计输入文本的总长度
        tsum = rs[0].sum()

        for m in tqdm(range(2, self.max_sep + 1)):
            print(f'正在挖掘{m}词短语...')
            rs.append([])
            for i in range(m):  # 生成所有可能的m字词，构造n元组
                for j in range(len(contents) - m + 1):
                    rs[m - 1].append(','.join(contents[j:j + m]))
            rs[m - 1] = pd.Series(rs[m - 1]).value_counts()  # 逐词统计
            rs[m - 1] = rs[m - 1][rs[m - 1] > self.min_count]  # 最小次数筛选
            tt = rs[m - 1][:]

            for k in range(m - 1):
                try:
                    qq = np.array(
                        list(
                            map(
                                lambda index: tsum * rs[m - 1][index] /
                                int(rs[m - 2 - k][','.join(
                                    index.split(',')[:m - 1 - k])]) / int(rs[
                                        k][','.join(
                                            index.split(',')[m - 1 - k:])]),
                                tt.index))) > self.min_support  # 最小支持度
                    tt = tt[qq]
                except Exception:
                    continue
            rt.append(tt.index)

        for i in tqdm(range(2, self.max_sep + 1)):
            print(f'正在筛选{i}词短语({len(rt[i - 2])})...')
            pp = []  # 保存所有的左右邻结果
            for j in range(len(contents) - i - 1):
                sp = ([
                    contents[j], ','.join(contents[j + 1:j + i + 1]),
                    contents[j + i + 1]
                ])
                pp.append(sp)
            pp = pd.DataFrame(pp).set_index(1).sort_index()  # 先排序，加快检索速度
            index = np.sort(np.intersect1d(rt[i - 2], pp.index))  # 作交集
            # 分别计算左邻和右邻信息熵
            index = index[np.array(
                list(
                    map(lambda s: calEnt(pd.Series(pp[0][s]).value_counts()),
                        index))) >= self.min_s]
            rt[i - 2] = index[np.array(
                list(
                    map(lambda s: calEnt(pd.Series(pp[2][s]).value_counts()),
                        index))) >= self.min_s]

        # 下面都是输出前处理
        for i in range(len(rt)):
            rs[i + 1] = rs[i + 1][rt[i]]
            rs[i + 1] = rs[i + 1].sort_values(ascending=False)

        # 返回词频字典
        key = list(
            chain(*[[''.join(ix.split(',')) for ix in list(elem.index)]
                    for elem in rs[1:]]))
        val = list(chain(*[[ix for ix in list(elem)] for elem in rs[1:]]))
        # value_dict =removeRedundant(dict(zip(key, val)))
        return dict(zip(key, val))


# 信息熵函数
def calEnt(sl):
    # -------------------------------------------------
    #    description: 计算信息熵
    #    param sl list
    #    return:
    # -------------------------------------------------
    sum_ent = np.sum(-((sl / sl.sum()).apply(np.log) * sl / sl.sum()))
    return sum_ent


def simJaccard(a, b):
    # -------------------------------------------------
    #    description: 利用改进的杰卡德距离计算相似度
    #    param a str/dict/list
    #    param b str/dict/list
    #    return:
    # -------------------------------------------------
    common = len(set(a).intersection(set(b)))
    length = min(len(set(a)), len(set(b)))
    return common / length


if __name__ == '__main__':
    passquit
    # print(cut('点击返回酒店和反对发射点发'))
    # contents = []
    # with codecs.open('/project/CORPUS/factjcjdb.txt',encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines):
    #         line = line.strip().split('\n')[0][1:-1]
    #         # line = list(pp.__re__(line))
    #         line = cut(line)
    #         contents += line
    # # print(contents)
    # ex = Extraction()
    # rs = ex.words(contents)
    # print(rs)