

# ----* tensorflow基本使用 *---- # 
# -实例2-
# 交互式会话


import tensorflow as tf 
sess = tf.InteractiveSession()      #创建交互式会话
a = tf.Variable( [1.0, 2.0] )       #创建变量数组
b = tf.constant( [3.0, 4.0] )       #创建常量数组
sess.run( tf.global_variables_initializer() )       #变量初始化
res = tf.add( a, b )        #创建加法操作
print( res.eval )           #执行操作并输出结果



