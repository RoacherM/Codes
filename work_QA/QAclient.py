#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/work_QA/QAclient.py
#    Description:     一个基于Flask框架的QA问答程序client
#    Author:     WY
#    Date: 2020/06/23
#    LastEditTime: 2020/06/24
# -------------------------------------------------

import requests

if __name__ == '__main__':

    url = 'http://0.0.0.0:5000/QA_system'
    # 外部访问使用http://172.16.3.220:5005/QA_system
    payload = {"text": '新车如何买保险'}
    response = requests.post(url, data=payload).text
    print(response.encode('utf-8').decode('unicode_escape'))
