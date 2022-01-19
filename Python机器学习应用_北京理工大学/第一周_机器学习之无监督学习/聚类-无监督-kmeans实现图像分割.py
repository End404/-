

# ---* 无监督-kmeans实现图像分割 *--- #


#***********************************************************
'''
    --实例描述--   
    目标：利用K-means聚类算法对图像像素点颜色进行聚类实现简单的图像分割.
    
    输出：同一聚类中的点使用相同颜色标记，不同聚类颜色不同.
    
    技术路线：sklearn.cluster.KMeans。
    
    
    --实例数据--  
    本实例中的数据可以是任意大小的图片，为了使效果更佳直观，可以采用区分度比较明显的图片。

'''
#***********************************************************




# 1-建立工程并导入相关包：
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans


# 2-加载图片并进行预处理： 
#加载训练数据.

def loadData(filePath):
    f = open(filePath, 'rb')   #以二进制形式打开文件。
    
    #以列表形式返回图片像素值
    data = []
    img = image.open(f)
    
    m, n = img.size    #获得图片大小。
    
    for i in range(m):     #将每个像素点RGB颜色处理到0--1。
        for j in range(n):    #范围内并存放进data。
            x, y, z = img.getpixel((i, j))
            data.append([x/255.0, y/256.0, z/256.0])
            
    f.close()
    
    return np.mat(data), m, n     #以矩阵形式返回data，以及图片大小。
#imgData, row, col = loadData('Kmeans/bull.jpg')    #加载数据。
imgData, row, col = loadData('bull.jpg')    #加载数据。


# 3-加载Kmeans聚类算法： 
Km = KMeans(n_clusters = 3)    #指定聚类中心的个数为3。 


# 4-对像素带你进行聚类并输出：
#依据聚类中心，对属于同一聚类的点使用同样的颜色进行标记。

#聚类获得每个像素所属的类别。
label = Km.fit_predict(imgData)
label = label.reshape([row, col])

pic_new = image.new("L", (row, col))    #创建一张新的灰度图保存聚类后的结果。

#根据所属类别想图片中添加灰度值。
for i in range(row):
    for j in range(col):
        #pic_new.putpixel( (i, j),256 / (label[i][j]+1) )  #方法出错。
        pic_new.putpixel( (i, j), int(256 / (label[i][j]+1)) )
        
pic_new.save("result-bull-4.jpg", "JPEG")    #以JPEG格式保存图像。

