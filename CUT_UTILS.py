#!/opt/miniconda/bin/python python
# coding: utf-8

# """
# -------------------------------------------------
#    File Name：     CUT_UTILS.py
#    Description :   一个多进程文本分词程序，较为适合处理大型文本
#    Author :        WY
#    date：          2020/6/5
# -------------------------------------------------
# """

import re
import os
import time
import codecs
import pkuseg
import chardet
from multiprocessing import Pool, cpu_count

userwords = [
    line.strip().split(',')[0] for line in codecs.open(
        '/Users/wonbyron/Desktop/work/Codes/project/DICTS/user.txt', encoding='utf-8').readlines()
]
stopwords = [
    line.strip().split(',')[0] for line in codecs.open(
        '/Users/wonbyron/Desktop/work/Codes/project/DICTS/stop.txt', encoding='utf-8').readlines()
]
mustwords = [
    line.strip().split(',')[0] for line in codecs.open(
        '/Users/wonbyron/Desktop/work/Codes/project/DICTS/must.txt', encoding='utf-8').readlines()
]

pseg = pkuseg.pkuseg(user_dict=userwords)

# 常见正则表达式
# url网址
Urls = '''[a-zA-z]+://[^\s]*'''
# IP地址
IPs = '''\d+\.\d+\.\d+\.\d+'''
Banks = '''/^([1-9]{1})(\d{14}|\d{18})$/'''
# 中国大陆固定电话号码
Phones = '''(\d{4}-|\d{3}-)?(\d{8}|\d{7})'''
# 中国大陆手机号码
Mphones = '''1\d{10}'''
# 中国大陆邮编
Mails = '''[1-9]\d{5}'''
# 电子邮箱
Emails = '''\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*'''
# 中国大陆身份证号(18位或者15位)
IDs = '''\d{15}(\d\d[0-9xX])?'''
# 中国车牌号码
Plates = '''([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}
        (([0-9]{5}[DF])|(DF[0-9]{4})))|([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼
        使领A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]{1})'''
# QQ号码
QQs = '''/^[1-9]\d{4,9}$/'''
# 微信号码
Wechats = '''/^[a-zA-Z]{1}[-_a-zA-Z0-9]{5,19}$/'''


# 采用多进程的方式读取文本
def wrapper(corpuscut, fpath, p_start, p_end, fsize):
    return corpuscut.loadata_multi(fpath, p_start, p_end, fsize)


class corpuscut():
    """
    主要功能，切分从数据库导出的原始txt文本，中间加空格并保存成txt
    """
    def __init__(self, fpath, spath=None, workers=None, coding=None):
        if not os.path.isfile(fpath):
            raise Exception(fpath + "file not exists!")
        self.f = fpath
        self.s = spath
        self.fsize = os.path.getsize(fpath)
        self.workers = workers if workers is not None else cpu_count() * 8
        self.cont = []
        if coding is None:
            with open(fpath, "rb") as fp:
                coding = chardet.detect(fp.read(10000))["encoding"]
        self.coding = coding

    def run(self):
        t0 = time.time()
        if self.workers == 1:
            self.loadata_single(self.f, self.fsize)
        else:
            res = []
            pool = Pool(self.workers)
            for i in range(self.workers):
                p_start = self.fsize * i // self.workers
                p_end = self.fsize * (i + 1) // self.workers
                args = [self, self.f, p_start, p_end, self.fsize]
                rs = pool.apply_async(func=wrapper, args=args)
                res.append(rs)
            pool.close()
            pool.join()
            for rs in res:
                self.cont.extend(rs.get())
        if self.s:
            with open(self.s, 'w', encoding='utf-8') as fp:
                for cont in self.cont:
                    r = ' '.join(cont) + '\n'
                    fp.writelines(r)
        cost = time.time() - t0
        cost = "{:.1f} seconds".format(cost) if cost < 60 else humantime(cost)
        size = humansize(self.fsize)
        tip = "\nFile size: {}. Workers: {}. Cost time: {}"
        print(tip.format(size, self.workers, cost))
        self.cost = cost + "s"

    def loadata_multi(self, fpath, p_start, p_end, fsize):
        # -------------------------------------------------
        #    description: 多进程切词程序，每次切分p_start-->p_end部分的文本，所有参数已指定好了
        #    param fpath 原始语料路径
        #    param p_start  开始读取的位置
        #    param p_end 终止读取的位置
        #    param fsize 文本大小
        #    return:
        # -------------------------------------------------
        contents = []
        with codecs.open(fpath, 'rb') as fp:
            if p_start:
                fp.seek(p_start - 1)
                while b"\n" not in fp.read(1):
                    pass
            t0 = time.time()
            while True:
                line = fp.readline()
                if line:
                    line = cut(line.decode(self.coding).strip().split('\n')[0])
                    contents.append(line)
                pos = fp.tell()
                if p_start == 0:
                    processbar(pos, p_end, fpath, fsize, t0)
                if pos >= p_end:
                    break
        return contents

    def loadata_single(self, fpath, fsize):
        # -------------------------------------------------
        #    description: 单进程切词程序，超过100M的文本建议使用多进程方式读取
        #    param fpath 原始语料路径
        #    param fsize 原始语料大小
        #    return:
        # -------------------------------------------------
        start = time.time()
        with codecs.open(fpath, 'rb') as fp:
            for line in fp:
                if line:
                    line = cut(line.decode(self.coding).strip().split('\n')[0])
                    self.cont.append(line)
                processbar(fp.tell(), fsize, fpath, fsize, start)


def dsre(text,
         without_Urls=True,
         without_IPs=False,
         without_Banks=True,
         without_Phones=True,
         without_Mphones=True,
         without_Mails=True,
         without_Emails=True,
         without_IDs=True,
         without_Plates=True,
         without_QQs=True,
         without_Wechats=True):
    # -------------------------------------------------
    #    description: 警情相关的正则表达式，当without某字段为真时，输出的文本则不包含该字段
    #    param ..
    #    return:
    # -------------------------------------------------
    if without_Urls:
        text = re.sub(Urls, '', text)
    if without_IPs:
        text = re.sub(IPs, '', text)
    if without_Banks:
        text = re.sub(Banks, '', text)
    if without_Phones:
        text = re.sub(Phones, '', text)
    if without_Mphones:
        text = re.sub(Mphones, '', text)
    if without_Mails:
        text = re.sub(Mails, '', text)
    if without_Emails:
        text = re.sub(Emails, '', text)
    if without_IDs:
        text = re.sub(IDs, '', text)
    if without_Plates:
        text = re.sub(Plates, '', text)
    if without_QQs:
        text = re.sub(QQs, '', text)
    if without_Wechats:
        text = re.sub(Wechats, '', text)
    # 只保留数字字母及下划线
    text = re.sub('\W+', ' ', text).replace('_', ' ')
    text = re.sub(
        '[\工号\d+号|\d+楼|\d+村|\d+幢|\d+巷|\d+室|\d+岁|\d+年|\d+月|\d+日|\d+时|\d+分|\d+秒|\d+点多|\d+时许|\d+日许|\d+多元|\d+元|\d+左右|\d+点钟|\d+点半|\d+日早]+',
        ' ', text)
    text = re.sub('[A-Za-z]+', ' ', text)
    text = re.sub('[\d]+', ' ', text)
    return text


def cut(texts):
    # -------------------------------------------------
    #    description: 切词器，
    #    param texts 输入为一则字符串/字符串列表
    #    return:
    # -------------------------------------------------
    sentences = []
    if isinstance(texts, list):
        for text in texts:
            segs = []
            words = pseg.cut(dsre(text))
            for word in words:
                if word not in stopwords or word in mustwords:
                    segs.append(word)
            if len(segs) > 1:
                sentences.append(segs)
    else:
        words = pseg.cut(dsre(texts))
        # print(words)
        for word in words:
            if word not in stopwords or word in mustwords:
                sentences.append(word)
        if len(sentences) > 1:
            return sentences
    return sentences


def humansize(size):
    # -------------------------------------------------
    #    description: 将文件的大小转成带单位的形式，主要用于绘制进度条
    #    param size 文件大小
    #    return:
    # -------------------------------------------------
    units = ["B", "KB", "M", "G", "T"]
    for unit in units:
        if size < 1024:
            break
        size = size // 1024
    return "{} {}".format(size, unit)


def humantime(seconds):
    # -------------------------------------------------
    #    description: 将秒数转成00:00:00的形式
    #    param seconds 所耗时间
    #    return:
    # -------------------------------------------------
    h = m = ""
    seconds = int(seconds)
    if seconds >= 3600:
        h = "{:02}:".format(seconds // 3600)
        seconds = seconds % 3600
    if 1 or seconds >= 60:
        m = "{:02}:".format(seconds // 60)
        seconds = seconds % 60
    return "{}{}{:02}".format(h, m, seconds)


def processbar(pos, p_end, fpath, fsize, t):
    # -------------------------------------------------
    #    description: 打印进度条 形式为a.txt, 50.00% [=====     ] 1/2 [00:01<00:01]
    #    param p_end
    #    param fpath
    #    param fsize
    #    param t
    #    return:
    # -------------------------------------------------
    """打印进度条
    just like:
    a.txt, 50.00% [=====     ] 1/2 [00:01<00:01]
    """
    percent = min(pos * 10000 // p_end, 10000)
    done = "=" * (percent // 1000)
    half = "-" if percent // 100 % 10 > 5 else ""
    tobe = " " * (10 - percent // 1000 - len(half))
    tip = "{}{}, ".format("\33[?25l", os.path.basename(fpath))  # 隐藏光标
    past = time.time() - t
    remain = past / (percent + 0.01) * (10000 - percent)
    processbar_fmt = "\r{}{:.1f}% [{}{}{}] {:,}/{:,} [{}<{}]"
    bar_content = processbar_fmt.format(
        tip,
        percent / 100,
        done,
        half,
        tobe,
        min(pos * int(fsize // p_end + 0.5), fsize),
        fsize,
        humantime(past),
        humantime(remain),
    )
    print(bar_content, end="")
    if percent == 10000:
        print("\33[?25h", end="")  # 显示光标


if __name__ == '__main__':
    # cp = corpuscut('/project/CORPUS/factjcjdb.txt','/project/CORPUS/corpus.txt',workers=256)
    # cp.run()
    pass