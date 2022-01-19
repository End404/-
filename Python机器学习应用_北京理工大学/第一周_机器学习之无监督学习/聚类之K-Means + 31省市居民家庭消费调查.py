
# --- 聚类之K-Means + 31省市居民家庭消费调查 --- #

# 1999年全国31个省份城镇居民家庭平均每人全年消费性支出的八个主要变量数据，这八个变量分别是：食品、衣着、家庭设备用品及服务、医疗.


'''
    k-means算法以k为参数，把n个对象分成k个簇，使簇内具有较高的相似度，而簇间的相似度较低。 
        其处理过程如下： 
        1.随机选择k个点作为初始的聚类中心； 
        2.对于剩下的点，根据其与聚类中心的距离，将其归入最近的簇 
        3.对每个簇，计算所有点的均值作为新的聚类中心 
        4.重复2、3直到聚类中心不再发生改变

                技术路线：sklearn.cluster.Kmeans 
'''

#1-建立工程，导入sklearn相关包：
import numpy as np 
from sklearn.cluster import KMeans 


#2-加载数据，创建K-means算法实例，并进行训练，获得标签：
#if__name__ == '__main__': 
if __name__ == '__main__':
    data, cityName = loadData( '31省市居民家庭消费水平-city.txt' )  #1、利用loadData方法读取数据.
    km = KMeans( n_clusters = 3 )     #2、创建实例。
    label = km.fit_predict( data )    #3、调用Kmeans()fit_predict()方法进行计算.
    expenses = np.sum( km.cluster_centers_, axis = 1 )
    
    #打印输出 expenses:
    '''
            将城市按label分成设定的簇。 
            将每个簇的城市输出。 
            将每个簇的平均输出。
    '''
        
    CityCluster = [ [], [], [] ]
    for i in range( len(cityName) ): 
        CityCluster[ label[i] ].append( cityName[i] ) 
    for i in range( len( CityCluster ) ): 
        print( "Expenses:%.2f" % expenses[i] )
        print( CityCluster[i] )
        

def loadData( filePath ): 
    
    
    ''' # ---读取文件---：
    
    .read() 每次读取整个文件，它通常用于将文件内容放到 一个字符串变量中 
    .readlines() 一次读取整个文件（类似于 .read() ) 
    .readline() 每次只读取一行，通常比 .readlines() 慢得 多。仅当没有足够内存可以一次读取整个文件时，才应 该使用 .readline()。
    
    '''
    
    fr = open( filePath, 'r+' )   #r+: 读写打开一个文本文件.
    lines = fr.readlines( )      # ---读取文件---。
    retData = [ ]        #用来存储城市的各项消费信息.
    retCityName = [ ]    #用来存储城市名字.
    
    for line in lines:
        items = lines.strip( ).split(",") 
        retCityName.append( items[0] )
        retData.append( [ fload (ltems[i] for i in range(1, len(items)) ) ] )
        
    for i in range( 1, len(items) ):
        return retData, retCityName     #返回值：返回城市名称，以及该城市的各项消费信息.
    
