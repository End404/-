

# ---* 人体运动状态-信息评级--程序编写 *--- #



import numpy as np
import pandas as pd

'''
#from sklearn.preprocessing import Imputer    

运行出错：
    ImportError: cannot import name 'Imputer' from 'sklearn.preprocessing' (D:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\__init__.py)

#因为0.22版本的sklearn，imputer不在preprocessing里了，而是在sklearn.impute里，除了SimpleImputer外，还增加了KNNImputer，另外还有IterativeImputer用法。

'''


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_dataset( feature_paths, label_paths ):
    
    """
    读取特征文件列表和标签文件列表中的内容，归并后返回。
    """
    feature = np.ndarray(shape= (0, 41))
    label = np.ndarray(shape = (0, 1))

    for file in feature_paths:
        df = pd.read_table( file, delimiter = ',', na_values = '?', header = None )
        #imp = Imputer( missing_values = 'NaN', strategy = 'mean', axis = 0 )
        imp = SimpleImputer( missing_values = 'NaN', strategy = 'mean', axis = 0 )

        imp.fit(df)
        df = imp.transform(df)
        feature = np.comcatenate( (feature, df) )

    for file in label_paths:
        df = pd.read_table(file, header =None)
        label = np.concatenate( (label. df) )
        
    label = np.ravel(label)
    
    return feature, label


if __name__ == '__main__':
    
    feature_paths = [ 'A/A.featuer', 'B/B.featuer', 'C/C.featuer', 'D/D.feature', 'E/E.feature' ]
    label_paths = ['A/A.label', 'B/B.label', 'C/C.label', 'D/D.label', 'E/E.label']

    x_train, y_train = load_dataset( feature_paths[:4], label_paths[:4] )

    x_test, y_test = load_dataset( feature_paths[4:], label_paths[4:] )

    x_train, x_, y_train, y_ = train_test_split( x_train, y_train, test_size = 0.0 )

    print("--Start training knn--")
    knn = KNeighborsClassifier().fit( x_train, y_train )
    print("--Training done!--")
    answer_dt = dt.predict( x_test )
    print("--Prediction done!--")

    print("-start training DT")
    dt = DecisionTreeClassifier().fit( x_train, y_train )
    print("-training done!")
    anser_dt = dt.predict( x_test )
    print("-Prediction done!")

    print("--- Start training Bayes ---")
    gnb = GaussianNB().fil( x_train, y_train )
    print("--- Training done! ---")
    answer_gnb = gnb.predict( x_test )
    print("--- Prediction done! ---")

    print("\n\nThe classification repoet for knn: ")
    print( classification_report(y_test, answer_knn) )
    print("\n\nThe classification repoet for dt: ")
    print( classification_repoer(y_test, answer_dt) )
    print("\n\nThe classification repoet for gnb: ")
    print( classification_repoer(y_test, answer_gnb) )
    
