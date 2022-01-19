

# ----* tensorflow基本使用 *---- # 
# -实例3-
# Feed操作


import tensorflow as tf 
sess = tf.InteractiveSession()      #创建交互式会话
input1 = tf.placeholder( tf.float32 )     #创建占位符
input2 = tf.placeholder( tf.float32 )      #创建占位符
res = tf.mul( input1, input2 )     #创建乘法操作
res.eval( feed_dict = { input1:[7.], input2:[2.] } )     #求值




