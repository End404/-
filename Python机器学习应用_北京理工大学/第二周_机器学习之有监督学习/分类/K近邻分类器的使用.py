

# -*- K近邻分类器的使用 -*-


"""

    --KNN：通过计算待分类数据点，与
已有数据集中的所有数据点的距离。取
距离最小的前K个点，根据“少数服从
多数“的原则，将这个数据点划分为出
现次数最多的那个类别。

#-------------------------------------------------------

    -在sklearn库中，可以使用sklearn.neighbors.KNeighborsClassifier
创建一个K近邻分类器，主要参数有：
    • n_neighbors：用于指定分类器中K的大小(默认值为5，注意与
kmeans的区别)
    • weights：设置选中的K个点对分类结果影响的权重（默认值为平均
权重“uniform”，可以选择“distance”代表越近的点权重越高，
或者传入自己编写的以距离为参数的权重计算函数）

    • algorithm：设置用于计算临近点的方法，因为当数据量很大的情况
下计算当前点和所有点的距离再选出最近的k各点，这个计算量是很
费时的，所以（选项中有ball_tree、kd_tree和brute，分别代表不
同的寻找邻居的优化算法，默认值为auto，根据训练数据自动选择）


"""


#创建一组数据X和它对应的标签y: 
X = [[0], [1], [2],[3]]
y = [0, 0, 1, 1]

#导入K近邻分类器.
from sklearn.neighbors import KNeighborsClassifier

#参数 n_neighbors 设置为 3，即使用最近的3个邻居作为分类的依据，其他参数保持默认值，并将创建好的实例赋给变量 neigh。
neigh = KNeighborsClassifier(n_neighbors = 3)

#调用 fit() 函数，将训练数据 X 和 标签 y 送入分类器进行学习。
neigh.fit(X, y) 

#调用 predict() 函数，对未知分类样本 [1.1] 分类，可以直接并将需要分类的数据构造为数组形式作为参数传入，得到分类标签作为返回值。
print(neigh.predict([[1.1]])) 
