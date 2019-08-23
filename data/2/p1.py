# -*- coding: utf-8 -*-
# @创建时间 : 21/8/2018
# @作者    : worry1613(549145583@qq.com)
# GitHub  : https://github.com/worry1613
# @CSDN   : http://blog.csdn.net/worryabout/
import jieba
import pandas as pd
import re
import jieba.posseg as pseg
import multiprocessing
import warnings

warnings.filterwarnings('ignore')
# 加停用词库
import codecs

stopwords = codecs.open('../../stop.dict', 'r', encoding='utf8').readlines()
stopwordlist = [w.strip() for w in stopwords]
# 结巴分词后的停用词性 [标点符号、连词、助词、副词、介词、时语素、‘的’、数词、方位词、代词]
stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']


def tokenization(doc, sw):
    """
    分词
    :param doc:         文本内容
    :param sw:          停用词列表
    :return:            分词后的列表
    """
    result = []
    text = re.findall('[\u4e00-\u9fa5]', doc)  # 只取汉字
    words = pseg.cut(''.join(text))
    for word, flag in words:
        if flag not in stop_flag and word not in sw:
            result.append(word)
    return ' '.join(result)


# 读取文件
neg = 'neg.txt'
pos = 'pos.txt'
df_pos = pd.read_csv(pos, header=None, skiprows=-1, sep="    ", engine='python', names=['type', 'content'])
df_neg = pd.read_csv(neg, header=None, skiprows=-1, sep="    ", engine='python', names=['type', 'content'])
df_pos.index += df_neg.shape[0]
df = pd.concat([df_neg, df_pos]).drop_duplicates()  # 合并，去重
df = df.dropna()  # 删除含有空数据的行

cpu_count = multiprocessing.cpu_count()
jieba.enable_parallel(cpu_count)

# 分词保存文件
df['words'] = [tokenization(row.content, stopwordlist) for row in df.itertuples()]
dfn = df[['type', 'words']]
dfn = dfn.loc[dfn['words'].str.len() > 1]  # 去除空内容
dfn.to_csv('words_type.csv', sep='\t', index_label='_id')  # ,index=False
dfn['words'].to_csv('words_w2v.txt', index=False)  # w2v语料库
