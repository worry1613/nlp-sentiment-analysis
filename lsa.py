# -*- coding: utf-8 -*-
# @创建时间 : 21/3/2018
# @作者    : worry1613(549145583@qq.com)
# GitHub  : https://github.com/worry1613
# @CSDN   : http://blog.csdn.net/worryabout/

from optparse import OptionParser
import pandas as pd
import logging

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as sk_model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--in', type=str, help='语料库文件', dest='corpusfile')
    parser.add_option('-l', '--label', type=str, help='标签列名', dest='label')
    parser.add_option('-d', '--data', type=str, help='数据列名', dest='train')
    options, args = parser.parse_args()

    _cfile = './/data//1//words_type.csv'
    _label = 'type'
    _data = 'words'
    corpusfile = _cfile if not options.corpusfile else options.corpusfile
    label = _label if not options.label else options.label
    data = _data if not options.train else options.train

    try:
        df = pd.read_csv(corpusfile,sep='\t')
    except Exception as e:
        logging.error(corpusfile+' 语料库文件名路径错误！')
        exit()

    try:
        label_ = df[label]
    except Exception as e:
        logging.error(label + ' 标签列名错误！')ç
        exit()

    try:
        data_ = df[data]
    except Exception as e:
        logging.error(data + ' 数据列名错误！')
        exit()

    #tfidf模型
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 4),
        # max_features=150000
    )
    # tfidf.fit(data_)
    x_train = tfidf.fit_transform(data_)

    lsa = TruncatedSVD(n_components=300, n_iter=10, random_state=42)
    # lsa.fit(x_train)
    x_train = lsa.fit_transform(x_train)

    model = GaussianNB()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label_, scoring='precision', cv=10, n_jobs=1)
    print('GaussianNB 交叉验证结果:', accs,'均值：', accs.mean())

    model = LogisticRegression()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label_, scoring='precision', cv=10, n_jobs=1)
    print('LogisticRegression 交叉验证结果:', accs,'均值：', accs.mean())

    model = LinearSVC()
    accs = sk_model_selection.cross_val_score(model, x_train, y=label_, scoring='precision', cv=10, n_jobs=1)
    print('LinearSVC 交叉验证结果:', accs, '均值：', accs.mean())










