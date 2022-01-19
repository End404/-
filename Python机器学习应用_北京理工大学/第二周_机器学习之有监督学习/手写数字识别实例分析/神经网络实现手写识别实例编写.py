

# -*- 神经网络实现手写识别实例编写 -*- # 



"""
    --- 任务介绍 ---

        手写数字识别是一个多分类问题，共有10个分类，每个手写数字图像的类别标签是0~9中的其中一个数。

        任务：利用sklearn来训练一个简单的全连接神经网络，
            即多层感知机（Multilayer perceptron，MLP）用于识别数据集DBRHD的手写数字。
        
"""



#步骤1：建立工程并导入sklearn包.

import numpy as np
from os import listdir    #使用listdir模块，用于访问本地文件。
from sklearn.neural_network import MLPClassifier 


#步骤2：加载训练数据.

# 1）在sklearnBP.py文件中，定义img2vector函数，将加载的32*32的图片矩阵展开成一列向量。
def img2vector(fileName):
    retMat = np.zeros( [1024], int )     #定义返回的矩阵，大小位1*1024。
    fr = open(fileName)        #打开包含32*32大小的数字文件。
    lines = fr.readlines()     #读取文件的所有行。
    for i in range(32):        #遍历文件所有行。
        for j in range(32):    #并将01数字存放在retMat中。
            retMat[i*32+j] = lines[i][j]
    return retMat

# 2）在sklearnBP.py文件中定义加载训练数据的函数readDataSet，并将样本标签转化为one-hot向量.
def readDataSet(path):
    fileList = listdir( path )     #获取文件夹下的所有文件。
    numFiles = len(fileList)       #统计需要读取的文件的数目。
    dataSet = np.zeros([numFiles, 1024], int )      #用于存放所有的数字文件。
    hwLabels = np.zeros([numFiles, 10])     #用于存放对应的标签one-hot.
    for i in range(numFiles):     #遍历所有的文件。
        filePath = fileList[i]     #获取文件名称/路径
        digit = int( filePath.split('_')[0] )     #通过文件名获取标签。
        hwLabels[i][digit] = 1.0                  #对应的one-hot标签置1。        dataSet[i] = img2vector( path + '/' + filePath )
        dataSet[i] = img2vector(path + '/' + filePath)     #读取文件内容。
    return dataSet, hwLabels

# 3）在sklearnBP.py文件中，调用readDataSet和img2vector函数加载数据，将训练的图片存放在train_dataSet中，对应的标签则存在train_hwLabels中
train_dataSet, train_hwLabels = readDataSet( 'trainingDigits' )


#步骤3：训练神经网络

# 1）在sklearnBP.py文件中，构建神经网络：设置网络的隐藏层数、各隐
# 藏层神经元个数、激活函数、学习率、优化方法、最大迭代次数。
#- 设置含100个神经元的隐藏层。
#- hidden_layer_sizes 存放的是一个元组，表示第i层隐藏层里神经元的个数. 
#- 使用logistic激活函数和adam优化方法，并令初始学习率为0.0001. 
clf = MLPClassifier( hidden_layer_sizes = (100, ),
                     activation = 'logistic', solver = 'adam', 
                     learning_rate_init= 0.0001, max_iter = 2000 )

# 2）在sklearnBP.py文件中，使用训练数据训练构建好的神经网络
#- fit函数能够根据训练集及对应标签集自动设置多层感知机的输入与输出层的神经元个数。
#- 例如train_dataSet为n*1024的矩阵，train_hwLabels为n*10矩阵，
#   则fit函数将MLP的输入层神经元个数设为1024，输出层神经元个数为10：
clf.fit(train_dataSet, train_hwLabels)


#步骤4：测试集评价.

# 1）在sklearnBP.py文件中，加载测试集
dataSet, hwLabels = readDataSet('testDigits')

# 2）使用训练好的MLP对测试集进行预测，并计算错误率：
res = clf.predict(dataSet)    #对测试集进行预测。
error_num = 0                 #统计预测错误的数目。
num = len(dataSet)            #测试集的数目。
for i in range(num):         #遍历预测结果。
    #比较长度为10的数组，返回包含01的数组，0为不同，1为相同. 
    #若预测结果与真实结果相同，则10个数字全为1，否则不全为1. 
    if np.sum( res[i] == hwLabels[i] ) < 10:
        error_num += 1
print( "--Total num: ", num, '\n' " --Wrong num: ", \
       error_num, '\n' " --WrongRate: ", error_num / float(num) )

