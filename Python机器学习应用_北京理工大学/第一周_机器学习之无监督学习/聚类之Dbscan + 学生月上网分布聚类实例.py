
# --- 聚类之Dbscan + 学生月上网分布聚类实例 --- #


'''
    ---DBSCAN方法及应用DBSCAN密度聚类---
        DBSCAN算法是一种基于密度的聚类算法：
         • 聚类的时候不需要预先指定簇的个数
         • 最终的簇的个数不定

        DBSCAN算法将数据点分为三类：
         • 核心点：在半径Eps内含有超过MinPts数目的点
         • 边界点：在半径Eps内点的数量小于MinPts，但是落在核心点的邻域内
         • 噪音点：既不是核心点也不是边界点的点

        DBSCAN算法流程：
         1.将所有点标记为核心点、边界点或噪声点；
         2.删除噪声点； 3.为距离在Eps之内的所有核心点之间赋予一条边；
         4.每组连通的核心点形成一个簇；
         5.将每个边界点指派到一个与之关联的核心点的簇中（哪一个核心点的半 径范围之内）。

'''


#2. 读入数据并进行处理
import numpy as np
#from sklearn.cluster import DBSCAN
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict( )
onlinetimes = [ ]
f = open( '学生月上网时间分布-TestData.txt' , encoding='utf-8')

for line in f:

    #读取每条数据中的mac地址， 开始上网时间，上网时长.
    mac = line.split(',')[2]
    onlinetime = int(line.split(',')[6])
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])

    #mac2id是一个字典： key是mac地址 value是对应mac地址的上网时长以及开 始上网时间.
    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append( (starttime, onlinetime) )
    else:
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]


real_X = np.array(onlinetimes).reshape((-1, 2))


#3-1. 上网时间聚类，创建DBSCAN算法实例，并进行训练，获得标签：

#调用BDSCAN方法进行训练，lbels为每个数据的蔟标签。
X = real_X[:, 0:1]
db = skc.DBSCAN( eps = 0.01, min_samples = 20 ).fit(X)
labels = db.labels_

#打印数据被记上的标签，计算标签位-1，即噪声数据的比例。
print('Labels（每个数据被划分的簇的分类）: ')
print(labels)
raito = len(labels[labels[:] == -1]) / len(labels)
print('Noise raito（噪声数据的比例）: ', format(raito, '.2%' ) )

print()

#计算蔟的个数并打印，评价聚类效果。
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters（簇的个数）: %d' % n_clusters_)
print("Silhoutte Coefficint（聚类效果评价指标）: %0.3f" % metrics.silhouette_score(X, labels))

print()

#打印各簇标号以及各簇内数据。
for i in range(n_clusters_):
    print("Clusrer（各簇标号以及各簇内数据）", i ,': ')
    print(list(X[labels == i] .flatten() ))

print()

#5-画直方图，分析实验结构。
import matplotlib.pyplot as plt
plt.hist(X, 24)

print()
print('=========================================================')



