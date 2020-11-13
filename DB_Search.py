#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/DB_Search.py
#    Description:     一个多进程数据库抽样方法的样例
#    Author:     WY
#    Date: 2020/06/11
#    LastEditTime: 2020/07/01
# -------------------------------------------------

import re
import os
import codecs
from tqdm import tqdm
from SQL_UTILS import MYSQL
from multiprocessing import Pool


def getlabels(sql):
    # -------------------------------------------------
    #    description: 用于查询案由表中的数据
    #    param sql  sql格式为 SELECT ****,**** FROM **** WHERE LEVEL = 0
    #    return: 按照案件级别，从表中选取编码和名称，返回一个包含类别的字典{label:[name,]}
    # -------------------------------------------------
    db = MYSQL()
    labeldict = {}
    results = db.select(sql)
    # print(df)
    for res in results:
        labeldict[res[0]] = [res[1]]
    print('标签采集完毕！')
    return labeldict


def sample(sqls, saveto):
    # -------------------------------------------------
    #    description: 一个单进程的采样程序
    #    param sqls 一组查询语句，sql的格式如下，目前只支持自动解析一个标签类别
    #               SELECT ****,**** FROM **** WHERE **** = {} ORDER BY RAND() LIMIT 1000
    #    param saveto 采样后的文本保存路径
    #    return:
    # -------------------------------------------------
    for sql in sqls:
        db = MYSQL()
        results = db.select(sql)
        code = re.findall('= \d{1,}', sql)[0][2:]
        print(code)
        content = []
        for res in tqdm(results):
            content.append(res[0] + '***' + res[1] + '***' + code)
        if saveto:
            if not os.path.exists(saveto):
                os.mkdir(saveto)
            fname = code + '.txt'
            with codecs.open(os.path.join(saveto, fname),
                             'w',
                             encoding='utf-8') as fp:
                for cont in content:
                    fp.writelines(cont + '\n')
        # return content


def splitask(sql, workers, codes):
    # -------------------------------------------------
    #    description: 切分子任务
    #    param sql sql默认格式为
    #              SELECT ****,**** FROM **** WHERE **** = {} ORDER BY RAND() LIMIT 1000
    #    param workers 指定线程数
    #    param codes 需要采集的标签id，从而补全sql的类别
    #    return:
    # -------------------------------------------------
    """切分子任务
    """
    parts = []
    for i in range(len(codes) // workers + 1):
        start = workers * i
        end = workers * (i + 1)
        part = []
        if isinstance(codes, dict):
            for code in list(codes.keys())[start:end]:
                part.append(sql.format(code))
        elif isinstance(codes, list):
            for code in codes[start:end]:
                part.append(sql.format(code))
        else:
            break
        parts.append(part)
    return parts


def main():

    sqlc = input(
        "Input your SQL for class:\ne.g.'SELECT ****,**** FROM **** WHERE LEVEL = 1'\n"
    )
    sqlf = input(
        "Input your SQL for contents:\ne.g.'SELECT ****,**** FROM **** WHERE **** = {} ORDER BY RAND() LIMIT 1000\n'"
    )
    workers = int(input('Set Workers:'))
    codes = getlabels(sqlc)
    pool = Pool(processes=workers)
    parts = splitask(sqlf, workers, codes)
    for part in parts:
        pool.apply_async(func=sample, args=(part, '/project/CORPUS/samples'))
    pool.close()
    pool.join()


if __name__ == '__main__':

    codes = ['****', '****', '****']
    sqlf = 'SELECT ****,**** FROM **** WHERE **** = {} ORDER BY RAND() LIMIT 1000'
    # codes = getlabels(sqlc)
    workers = 36
    pool = Pool(processes=workers)
    parts = splitask(sqlf, workers, codes)
    for part in parts:
        pool.apply_async(func=sample, args=(part, 'samples'))
    pool.close()
    pool.join()
    # main()