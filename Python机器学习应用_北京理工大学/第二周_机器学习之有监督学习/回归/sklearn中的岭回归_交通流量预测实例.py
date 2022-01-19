



# ---- sklearn中的岭回归_交通流量预测实例 ---- #


'''
    数据介绍：
        数据为某路口的交通流量监测数据，记录全年小时级别的车流量。

    实验目的：
        根据已有的数据创建多项式特征，使用岭回归模型代替一般的线性模型，对
    车流量的信息进行多项式回归。
                    
                    技术路线：sklearn.linear_model.Ridgefrom
                        sklearn.preprocessing.PolynomialFeatures


'''



#1. 建立工程，导入sklearn相关工具包：
import numpy as np 
from sklearn.linear_model import Ridge      #通过sklearn.linermodel加载岭回归方法。
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation     #加载交叉验证模块，加载matplotilib模块。
from sklearn.model_selection import cross_val_score  
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures      #用于创建多项式特征。


#2. 数据加载：
data = np.genfromtxt('data.txt')     #使用numpy的方法从txt文件中加载数据。
#data = np.genfromtxt('data.csv')
#data = np.loadtxt( 'data.csv' )
#plt.plot(data[:,4])      #使用plt展示车流量信息。


#3. 数据处理：
X = data[:,:4]    ##X用于保存0-3维数据，即属性。
y = data[:,4]    #y用于保存第4维数据，即车流量。
poly = PolynomialFeatures(6)    ##用于创建最高次数6次方的的多项式特征，多次试验后决定采用6次。
X = poly.fit_transform(X)      ##X为创建的多项式特征。


#4. 划分训练集和测试集：
#将所有数据划分为训练集和测试集，test_size表示测试集的比例，
#random_state是随机数种子。
#train_set_X, test_set_X, train_set_y, test_set_y = cross_validation.train_test_split(X, y, test_size = 0.3, random_state = 0) 
#train_set_X, test_set_X, train_set_y, test_set_y = cross_val_score.train_test_split(X, y, test_size = 0.3, random_state = 0) 
train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size = 0.3, random_state = 0) 


#5. 创建回归器，并进行训练：
clf = Ridge(alpha = 1.0, fit_intercept = True )     #接下来我们创建岭回归实例。
clf = clf.fit(train_set_X, train_set_y)     #调用fit函数使用训练集训练回归器。
clf = clf.score(test_set_X, test_set_y)#利用测试集计算回归曲线的拟合优度，clf.score返回值为0.7375。 #拟合优度，用于评价拟合好坏，最大为1，无最小值，当对所有输入都输出同一个值时，拟合优度为0。


#6. 画出拟合曲线：
start = 200     #接下来我们画一段200到300范围内的拟合曲线。
end = 300      
y_pre = clf.predict(X)     #是调用predict函数的拟合值。
time = np.arange(start, end) 

#展示真实数据（蓝色）以及拟合的曲线（红色）。
plt.plot(time, y[start:end], 'b', label = 'real') 
plt.plot(time, y_pre[start:end], 'r', label = 'predict')

plt.legend(loc = 'upper left')      #设置图例的位置。
plt.show()


