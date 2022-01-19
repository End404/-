
# -*- sklearn中的决策树 -*-

#***********************************************************************************

"""

    --决策树--是一种树形结构的分类器，通过顺序
询问分类点的属性决定分类点最终的类别。通常
根据特征的信息增益或其他指标，构建一颗决策
树。在分类时，只需要按照决策树中的结点依次
进行判断，即可得到样本所属类别。

------------------------------------------------

        --sklearn中的决策树--
        在sklearn库中，可以使用sklearn.tree.DecisionTreeClassifier创
建一个决策树用于分类，其主要参数有：
        • criterion ：用于选择属性的准则，可以传入“gini”代表基尼
系数，或者“entropy”代表信息增益。
        • max_features ：表示在决策树结点进行分裂时，从多少个特征
中选择最优特征。可以设定固定数目、百分比或其他标准。它的默认值是使用所有特征个数。
        
"""
#***********************************************************************************



#首先，我们导入 sklearn 内嵌的鸢尾花数据集：
from sklearn.datasets import load_iris 

#接下来，我们使用 import 语句导入决策树分类器，同时导入计算交叉验 证值的函数 cross_val_score。
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score 

#我们使用默认参数，创建一颗基于基尼系数的决策树，并将该决策树分类 器赋值给变量 clf。
clf = DecisionTreeClassifier()

#将鸢尾花数据赋值给变量 iris。
iris = load_iris() 

#这里我们将决策树分类器做为待评估的模型， iris.data鸢尾花数据做为特征， iris.target鸢尾花分类标签做为目标结果，通过设定cv为10，使用10折交叉验 证。得到最终的交叉验证得分
cross_val_score(clf, iris.data, iris.target, cv = 10) 

#以仿照之前 K近邻分类器的使用方法，利用 fit() 函数训练模型并使用 predict() 函数预测：
clf.fit(X, y)   #NameError: name 'X' is not defined
clf.predict(X)


#**************************************
'''
控制台运行结果：
clf.fit(X, y)
Out[22]: DecisionTreeClassifier()

clf.predict(X) 
Out[19]: array([0, 0, 1, 1])

---------------------------

clf.predict(x)

NameError: name 'x' is not defined
'''
#***************************************


