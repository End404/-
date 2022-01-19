
# --- 监督学习 上证指数涨跌预测 --- #
# -2- 


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

#from sklearn import cross_validation
 
'''
这是因为 sklearn 0.21.1 版本的已经移除 cross_validation 模块
从 sklearn.model_selection 模块直接导入 cross_val_score 即可
'''

data=pd.read_csv('000777.csv',encoding='gbk',parse_dates=[0],index_col=0)
data.sort_index(0,ascending=True,inplace=True)
 
dayfeature=150
featurenum=5*dayfeature
x=np.zeros((data.shape[0]-dayfeature,featurenum+1))
y=np.zeros((data.shape[0]-dayfeature))

''' 
for i in range(0,data.shape[0]-dayfeature):
    x[i,0:featurenum]=np.array(data[i:i+dayfeature] \
          [[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
    x[i,featurenum]=data.ix[i+dayfeature][u'开盘价']
 
for i in range(0,data.shape[0]-dayfeature):
    if data.ix[i+dayfeature][u'收盘价']>=data.ix[i+dayfeature][u'开盘价']:
        y[i]=1
    else:
        y[i]=0  
    --------------------------------------------------------
    了解到是因为在pandas的1.0.0版本开始，移除了Series.ix and DataFrame.ix 方法。
代替方法使用
    可以使用 iloc 代替 pandas.DataFrame.iloc        
'''
 
 
for i in range(0,data.shape[0]-dayfeature):
    x[i,0:featurenum]=np.array(data[i:i+dayfeature] \
          [[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
    x[i,featurenum]=data.iloc[i+dayfeature][u'开盘价']
 
for i in range(0,data.shape[0]-dayfeature):
    if data.iloc[i+dayfeature][u'收盘价']>=data.iloc[i+dayfeature][u'开盘价']:
        y[i]=1
    else:
        y[i]=0
        
clf=svm.SVC(kernel='rbf')
result = []
for i in range(5):
    x_train, x_test, y_train, y_test = \
                cross_val_score.train_test_split(x, y, test_size = 0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm classifier accuacy:")
print(result)