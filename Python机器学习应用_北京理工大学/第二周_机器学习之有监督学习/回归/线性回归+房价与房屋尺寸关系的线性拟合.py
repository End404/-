

# 线性回归+房价与房屋尺寸关系的线性拟合 -*-


"""
    --- 线性回归的应用 ----
    
    目标：对房屋成交信息建立回归方程，并依据回归方程对房屋价格进行预测
    
    技术路线：sklearn.linear_model.LinearRegression
    
    背景：与房价密切相关的除了单位的房价，还有房屋的尺寸。我们可以根
据已知的房屋成交价和房屋的尺寸进行线性回归，继而可以对已知房屋尺
寸，而未知房屋成交价格的实例进行成交价格的预测。


    为了方便展示，成交信息只使用了房屋的面积以及对应的成交价格。
    其中：
    • 房屋面积单位为平方英尺（ft2）房。
    • 屋成交价格单位为万。
    
"""



#-实验过程
#-使用算法：线性回归。
#-实现步骤：
#-1.建立工程并导入sklearn包。
#-2.加载训练数据，建立回归方程。
#-3.可视化处理。


import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model   #表示，可以调用sklearn中的linear_model模块进行线性回归。

#建立datasets_X和datasets_Y用来存储数据中的房屋尺寸和房屋成交价格。
datasets_X = []
datasets_Y = []

fr = open( 'prices.txt', 'r' )    #打开数据集所在文件prices.txt，读取数据。
lines = fr.readlines()    #一次读取整个文件。

for line in lines:    #逐行进行操作，循环遍历所有数据.
    items = line.strip().split(',')     #去除数据文件中的逗号.
    
    #将读取的数据转换为int型，并分别写入datasets_X和datasets_Y。
    datasets_X.append( int(items[0]) )
    datasets_Y.append( int(items[0]) )
    
length = len(datasets_X)     #求得datasets_X的长度，即为数据的总数。
datasets_X = np.array(datasets_X).reshape( [length, 1] )    #将datasets_X转化为数组，并变为二维，以符合线性回归拟合函数输入参数要求。
datasets_Y = np.array(datasets_Y)     #将datasets_Y转化为数组.

#以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图。
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange( minX, maxX ).reshape( [-1, 1] )

#调用线性回归模块，建立回归方程，拟合数据.
linear = linear_model.LinearRegression()
linear.fit( datasets_X, datasets_Y )


print ( 'Coefficients（查看回归方程系数）： ' , linear.coef_ )
print ( "intercept（查看回归方程截距）： ", linear.intercept_ )


plt.scatter( datasets_X, datasets_Y, color = 'red' )     #scatter函数用于绘制数据点，这里表示用红色绘制数据点.

plt.plot( X, linear.predict(X), color = 'blue' )     #plot函数用来绘制直线，这里表示用蓝色绘制回归线.

#plot函数用来绘制直线，这里表示用蓝色绘制回归线；xlabel和ylabel用来指定横纵坐标的名称。
plt.xlabel('Area')
plt.ylabel("Price")
plt.show()



