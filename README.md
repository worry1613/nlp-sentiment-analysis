# nlp-sentiment-analysis
中文自然语言处理情感分析，多算法，多数据集


## 数据集
### 1   
2W条书籍评价情感分析数据（正负）  
链接:https://pan.baidu.com/s/1sQvWhXngi7kHS_97AfXdJw  密码:e5h9  
### 2  
谭松波--酒店评论语料  
链接:https://pan.baidu.com/s/1N7VMCld0TuXCk5Htxz_LLA  密码:pgpo  

## 操作步骤   
1.处理数据文件格式  
2.分词  
3.特征工程，TFIDF，LSI，W2V  
4.算法，LR,SVM,BAYES  
5.效果评价  

## 如何用不用   
1.数据目录下的p1.py文件，生成分词后的词料库文件和word2vec语料库文件  
2.tfidf.py生成TFIDF特征并用3种算法(LR,SVM,BAYES)分类，10折交叉验证  
3.lsa.py生成LSA特征并用3种算法(LR,SVM,BAYES)分类，10折交叉验证  
4.w2v_model.py生成word2vec模型  
5.word2vector.py生成word2vec特征并用3种算法(LR,SVM,BAYES)分类，10折交叉验证  