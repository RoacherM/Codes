#!/opt/miniconda/bin/python
# coding: utf-8

# -------------------------------------------------
#    FilePath: /project/work_QA/QAserver.py
#    Description:     一个基于Flask框架的QA问答程序server
#    Author:     WY
#    Date: 2020/06/23
#    LastEditTime: 2020/06/23
# -------------------------------------------------

import sys
import flask
import pandas as pd
sys.path.append('.')
import tensorflow as tf
from annoy import AnnoyIndex
from SIM_UTILS import simBERT

# config path
csv = '/project/work_QA/BX8K.csv'
ann = '/project/work_QA/qa.ann'
# init flask method
app = flask.Flask(__name__)
model = simBERT()
global graph
graph = tf.get_default_graph()

questions = list(pd.read_csv(csv)['title'])
answers = list(pd.read_csv(csv)['reply'])


@app.route('/QA_system', methods=["POST", "GET"])
def predict():
    if flask.request.method == "POST":
        text = flask.request.form["text"]
        print('------问题------', text)
        with graph.as_default():
            Z = model.sent2vec(text)
        db = AnnoyIndex(len(Z[0]), 'angular')
        db.load(ann)
        aid = db.get_nns_by_vector(Z[0], 1)
        ans = answers[aid[0]]
    else:
        return "GET Method is Not Support!"
    return flask.jsonify(ans)


if __name__ == '__main__':
    print(
        'Loading QA engine methods,please wait until server has fully started!'
    )
    app.run(host='0.0.0.0')