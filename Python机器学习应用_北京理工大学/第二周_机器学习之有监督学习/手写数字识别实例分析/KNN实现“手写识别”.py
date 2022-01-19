

# ---* KNN实现“手写识别” *--- #


'''
       --- 任务介绍 ---
       
    手写数字识别是一个多分类问题，共有10个分类，每个手写数字图像的类别标签是0~9中的其中一个数。
    
        本实例利用sklearn来训练一个K最近邻（k-Nearest Neighbor，KNN）分类器，用于识别数据集DBRHD的手写数字。
      比较KNN的识别效果与多层感知机的识别效果。
'''



#步骤1：建立工程并导入sklearn包.
import numpy as np
from os import listdir
from sklearn import neighbors


#步骤2：加载训练数据.
# 1）在sklearnKNN.py文件中，定义img2vector函数，将加载的32*32的图片矩阵展开成一列向量.
def img2vector( fileName ):
    retMat = np.zeros( [1024], int )
    fr = open( fileName )
    lines = fr.readlines()
    for i in range( 32 ):
        for j in range( 32 ):
            retMat[i * 32 + j] = lines[i] [j]
    return retMat

# 2）在sklearnKNN.py文件中定义加载训练数据的函数readDataSet。
def readDataSet( path ):
    fileList = listdir( path )
    numFiles = len( fileList )
    dataSet = np.zeros( [numFiles, 1024], int )
    hwLabels = np.zeros( [numFiles] )
    for i in range( numFiles ):
        filePath = fileList[i]
        digit = int( filePath.split('_') [0] )
        hwLabels[i] = digit
        dataSet[i] = img2vector( path + '/' + filePath )
    return dataSet, hwLabels

# 3）在sklearnKNN.py文件中，调用readDataSet和img2vector函数加载数据，将训练的图片存放在train_dataSet中，对应的标签则存在train_hwLabels中.
train_dataSet, train_hwLabels = readDataSet( 'trainingDigits' )


#步骤3：构建KNN分类器.
# 1）在sklearnKNN.py文件中，构建KNN分类器：设置查找算法以及邻居点数量(k)值。
# KNN是一种懒惰学习法，没有学习过程，只在预测时去查找最近邻的点，数据集的输入就是构建KNN分类器的过程。
# 构建KNN时我们同时调用了fit函数。
knn = neighbors.KNeighborsClassifier( algorithm = 'kd_tree' , n_neighbors = 3 )
knn.fit( train_dataSet, train_hwLabels )


#步骤4：测试集评价.
# 1）加载测试集：
dataSet, hwLabels = readDataSet( 'testDigits' )

# 2）使用构建好的KNN分类器对测试集进行预测，并计算预测的错误率.
res = knn.predict(dataSet)
error_num = np.sum( res != hwLabels )
num = len( dataSet )
print( " --Total num： ", num, 
      '\n'
      " --Wrong num: ",error_num, 
      '\n'
      " --WrongRate: ", error_num / float(num) )




