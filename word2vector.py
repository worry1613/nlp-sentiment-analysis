# -*- coding: utf-8 -*-
# @创建时间 : 21/3/2018
# @作者    : worry1613(549145583@qq.com)
# GitHub  : https://github.com/worry1613
# @CSDN   : http://blog.csdn.net/worryabout/

from optparse import OptionParser
import pandas as pd
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as sk_model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--in', type=str, help='语料库文件', dest='corpusfile')
    parser.add_option('-l', '--label', type=str, help='标签列名', dest='label')
    parser.add_option('-d', '--data', type=str, help='数据列名', dest='train')
    parser.add_option('-m', '--model', type=str, help='w2v模型文件名', dest='model')
    options, args = parser.parse_args()

    parm_corpusfile = './/data//1//words_type.csv' if not options.corpusfile else options.corpusfile
    parm_model = './/data//1//w2v.bin' if not options.corpusfile else options.corpusfile
    parm_label = 'type' if not options.label else options.label
    parm_data = 'words' if not options.train else options.train

    logging.info('语料库文件:' + parm_corpusfile)
    logging.info('w2v模型文件名:' + parm_model)
    logging.info('标签列名:' + parm_label)
    logging.info('数据列名:' + parm_data)
    try:
        df = pd.read_csv(parm_corpusfile, sep='\t')
    except Exception as e:
        logging.error(parm_corpusfile + ' 语料库文件名路径错误！')
        exit()

    w2vmodel = None
    try:
        from gensim.models import word2vec

        w2vmodel = word2vec.Word2Vec.load(parm_model)
    except Exception as e:
        logging.error(parm_model + ' w2v模型文件名错误！')
        exit()

    if parm_label not in df.columns:
        logging.error(parm_label + ' 标签列名错误！')
        exit()
    if parm_data not in df.columns:
        logging.error(parm_label + ' 标签列名错误！')
        exit()

    ct = []
    for row in df[parm_data].iteritems():
        t = []
        for word in row[1].split(' '):
            if word in w2vmodel.wv.vocab:
                t.append(w2vmodel.wv[word])
        ct.append(np.mean(np.array(t), axis=0).tolist())

    df['w2v'] = ct
    df = df.dropna()  # 删除含有空数据的行
    logging.info('共%d数据，其中正数据%d，负数据%d ' %(len(df),len(df[df['type']==1]),len(df[df['type']==-1])))

    x_train = df['w2v'].tolist()
    label = df['type'].tolist()

    model = GaussianNB()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label, scoring='precision', cv=10, n_jobs=1)
    print('GaussianNB 交叉验证结果:', accs, '均值：', accs.mean())
    #
    model = LogisticRegression()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label, scoring='precision', cv=10, n_jobs=1)
    print('LogisticRegression 交叉验证结果:', accs,'均值：', accs.mean())

    model = LinearSVC()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label, scoring='precision', cv=10, n_jobs=1)
    print('LinearSVC 交叉验证结果:', accs, '均值：', accs.mean())
