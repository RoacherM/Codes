# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-07-01 10:38:22
LastEditTime: 2020-11-13 18:46:36
'''
#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/work_QA/QAsystem.py
#    Description:
#    Author:     WY
#    Date: 2020/06/13
#    LastEditTime: 2020/07/01
# -------------------------------------------------


import sys
import pandas as pd
from tqdm import tqdm
from annoy import AnnoyIndex
# 从上级文件中导入模块
sys.path.append('.')
from SIM_UTILS import simBERT


class Model(simBERT):
    """简单问答系统设计
    """
    def __init__(self,
                 csv='/Users/wonbyron/Desktop/work/Codes/work_QA/BX8K.csv',
                 ann='/Users/wonbyron/Desktop/work/Codes/work_QA/qa.ann'):
        super().__init__()
        try:
            self.questions = list(pd.read_csv(csv)['title'])
        except Exception:
            print('Cant locate column *title*')
        try:
            self.answers = list(pd.read_csv(csv)['reply'])
        except Exception:
            print('Cant locate column *reply* ')
        self.qa = ann

    def train(self):
        # -------------------------------------------------
        #    description: 训练
        #    return:
        # -------------------------------------------------
        print('Encoding Vectors...')
        Z = self.sent2vec(self.questions)
        print('Saving Vectors...')
        db = AnnoyIndex(len(Z[0]), 'angular')
        for i in tqdm(range(len(Z))):
            db.add_item(i, Z[i])
        db.build(10)
        db.save(self.qa)

    def predict(self, text):
        # -------------------------------------------------
        #    description: 预测
        #    param text str，文本
        #    return:
        # -------------------------------------------------
        Z = self.sent2vec(text)
        db = AnnoyIndex(len(Z[0]), 'angular')
        db.load(self.qa)
        return db.get_nns_by_vector(Z[0], 1)


def main():
    system = Model()
    while True:
        ques = input("输入要查询的语句，按'#q'退出，例如：\n如何交保险、车保、国外旅游保险等\n")
        if ques == '#q':
            break
        ans_id = system.predict(ques)
        print(f'您是问：{system.questions[ans_id[0]]}')
        sure = input('y/n:')
        if sure == 'y':
            print(f'很高兴为您解答：{system.answers[ans_id[0]]}')
            break
        else:
            print('我不知道你在说什么，能否描述详细一些。')


if __name__ == '__main__':

    # 输入新数据时，需要先进行向量编码
    # system = Model()
    # system.train()
    # 测试程序
    main()
