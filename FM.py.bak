# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:47:41 2018

@author: mengxiaoyu
"""

import tensorflow as tf  
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

#data load
iris=load_iris()
x=iris['data']
y=iris['target']

x,y=x[y!=2],y[y!=2]
for i in range(len(y)):
    if y[i]==0:
        y[i]=-1
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
n=4
k=20#特征数

#fm al
w0=tf.Variable(0.1)
w1=tf.Variable(tf.truncated_normal([n]))
w2=tf.Variable(tf.truncated_normal([n,k]))

x_=tf.placeholder(tf.float32,[None,n])
y_=tf.placeholder(tf.float32,[None])
batch=tf.placeholder(tf.int32)

w2_new=tf.reshape(tf.tile(w2,[batch,1]),[-1,n,k])
board_x=tf.reshape(tf.tile(x_,[1,k]),[-1,n,k])#-1:自动计算(none)，任意行,4列,k维

q=tf.square(tf.reduce_sum(tf.multiply(w2_new,board_x),axis=1))#按行求和再取平方
h=tf.reduce_sum(tf.multiply(tf.square(w2_new),board_x),axis=1)

y_fm=w0+tf.reduce_sum(tf.multiply(x_,w1),axis=1)+1/2*tf.reduce_sum(q-h,axis=1)

cost=tf.reduce_sum(tf.square(y_fm-y_))

batch_fl=tf.cast(batch,tf.float32)  
auc=tf.reduce_mean(tf.sign(tf.multiply(y_fm,y_)))         
train_op=tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)  
  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    for i in range(10000):  
        sess.run(train_op,feed_dict={x_:x_train,y_:y_train,batch:70})  
        print (sess.run(cost,feed_dict={x_:x_train,y_:y_train,batch:70}))  
    print (sess.run(auc,feed_dict={x_:x_test,y_:y_test,batch:30}))    
