

# 线性回归+房价与房屋尺寸关系的线性拟合 -*-
# -2-


#-实验过程
#-使用算法：线性回归。
#-实现步骤：
#-1.建立工程并导入sklearn包。
#-2.加载训练数据，建立回归方程。
#-3.可视化处理。



import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
 
 
# 读取数据集
datasets_X = []
datasets_Y = []
fr = open('prices.txt','r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
 
length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)
 
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])
 
 
linear = linear_model.LinearRegression()
linear.fit(datasets_X, datasets_Y)
 
# 图像中显示
plt.scatter(datasets_X, datasets_Y, color = 'red')
plt.plot(X, linear.predict(X), color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()



