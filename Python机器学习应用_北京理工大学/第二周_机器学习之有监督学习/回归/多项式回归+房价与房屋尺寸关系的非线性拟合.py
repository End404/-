


# -*- 多项式回归+房价与房屋尺寸关系的非线性拟合 -*-



"""
        ----多项式回归的应用----
        应用背景：我们在前面已经根据已知的房屋成交价和房屋的尺寸进行了线性回归，继而可以对已知房屋尺寸，而未知房屋成交价格的实例进行了成
        交价格的预测，但是在实际的应用中这样的拟合往往不够好，因此我们在此对该数据集进行多项式回归。
        
        目标：对房屋成交信息建立多项式回归方程，并依据回归方程对房屋价格进行预测。
                
                技术路线：sklearn.preprocessing.PolynomialFeatures

"""



import matplotlib.pyplot as plt 
import numpy as np 

#导入线性模型和多项式特征构造模块.
from sklearn import linear_model 
from sklearn.preprocessing import PolynomialFeatures 


#建立datasets_X和datasets_Y用来存储数据中的房屋尺寸和房屋成交价格。
datasets_X = []
datasets_Y = []

fr = open('prices.txt', 'r')     #打开数据集所在文件prices.txt，读取数据。
lines = fr.readlines()      #一次读取整个文件。


for line in lines:     #逐行进行操作，循环遍历所有数据.
    items = line.strip().split(',')      #去除数据文件中的逗号.
    datasets_X.append( int(items[0]) )    #将读取的数据转换为int型，并分别写入datasets_X和datasets_Y。
    datasets_Y.append( int(items[1]) )
    
length = len(datasets_X)     #求得datasets_X的长度，即为数据的总数。
datasets_X = np.array(datasets_X).reshape([length, 1])    #将datasets_X转化为数组，并变为二维，以符合线性回归拟合函数输入参数要求。
datasets_Y = np.array(datasets_Y)     #将datasets_Y转化为数组.


#以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图。
minX =min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX, maxX).reshape([-1, 1])


#degree=2表示建立datasets_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和datasets_Y之间的映射关系（即参数）。
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_X)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit( X_poly, datasets_Y )

#lin_reg_2.fit(X_poly, datasets_X)


#scatter函数用于绘制数据点，这里表示用红色绘制数据点；plot函数用来绘制回归线，同样这里需要先将X处理成多项式特征；xlabel和ylabel用来指定横纵坐标的名称。
plt.scatter(datasets_X, datasets_Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()




