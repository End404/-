
# --- 监督学习 上证指数涨跌预测 --- #

"""
    数据介绍：
    网易财经上获得的上证指数的历史数据，爬取了20年的上证指数数据。
    
    实验目的：
    根据给出当前时间前150天的历史数据，预测当天上证指数的涨跌。
            
    技术路线：sklearn.svm.SVC
    
    
    数据实例：中核科技1997年到2017年的股票数据部分截图，红框部分为选取的特征值
    
"""


import pandas as pd 
import numpy as np 
from sklearn import svm 
from sklearn.model_selection import cross_val_score

#from sklearn import cross_validation 
'''
    这是因为 sklearn 0.21.1 版本的已经移除 cross_validation 模块
从 sklearn.model_selection 模块直接导入 cross_val_score 即可
'''


data = pd.read_cvs('000777.cvs', encoding = 'gdk', parse_parse_dates = [0], index_col =0)
data.sort_index(0, ascending = True, inplace = False)

dayfeature = 150
fraturenum = 5 * dayfeatuer 
x = np.zeros( (data.shape[0] - dayfeature, featurenum + 1) )
y = np.zeros( (data.shape[0] - dayfeature) )


for i in range(0, data.shape[0] - dayfeature):
    x[ i, 0:featurenum ] = np.array( data[i : i + dayfeature] [ [u'收盘价', u'最高价', u'最低价', u'开盘价', u'成交量'] ]) .reshape( (1, featurenum) )
    
    x[ i, featurenum ] = data.ix[ i + dayfeature ] [u'开盘价']
    
for i in range( 0, data.shape[0] - dayfeature ): 
    if data.ix[ i + dayfeature ] [ u'收盘价' ] >= data.ix[i + dayfeature] [u'开盘价']: 
        y[i] = 1
    else: 
        y[i] = 0 
        
        
clf = svm.SVM( kernel = 'rbf' )

result = []

for i in range( 5 ): 
    x_train , x_test, y_train, y_test = cross_val_score.train_test_split( x, y, test_size = 20 )
    
    clf.fit(x_train, y_train)
    
    result.append( np.mean(y_test == clf.predict(x_test)) )
    
print("Svm classifrier accuacy.") 

print( result )




    