import tensorflow.compat.v1 as tf #symbolic math library used for machine learning
tf.disable_v2_behavior()
import pandas as pd #data manipulation and analysis
import sklearn.model_selection #machine learning library for python
import numpy as np #converts things into arrays easier to understand
##tf.compat.v1.disable_eager_execution()
##import tf.compat.v1 as tf

df=pd.read_csv('/Users/sasankgamini/Desktop/cancer.csv')
inputs=df.drop(['class','id'],axis=1) #columns is 1 and 0 is rows
inputs=np.array(inputs) #using numpy so it arranges the data so it's easier to read
outputs=df['class']
outputs=np.array(outputs)

##print(inputs)
##print(outputs)
##print(df['class'])
Xtraindata, Xtestdata, Ytraindata, Ytestdata = sklearn.model_selection.train_test_split(inputs,outputs, test_size = 0.2) #splitting into testing and training, need more training so we chose test size to be 20%
##print(len(Ytraindata))
print(Xtraindata)
print(Ytraindata)
print(Xtestdata)
print(Ytestdata)

count=0
trainplaceholder=tf.placeholder(tf.float32,[None,9])
testplaceholder=tf.placeholder(tf.float32,[9])
distance=tf.reduce_sum(tf.abs(trainplaceholder-testplaceholder),reduction_indices=1)
index=tf.arg_min(distance,0)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    count=0
    for n in range(0,140,1):
##        print(Xtestdata[n])
        distanceval=sess.run(distance,{trainplaceholder:Xtraindata,testplaceholder:Xtestdata[n]})
        indexval=sess.run(index,{trainplaceholder:Xtraindata,testplaceholder:Xtestdata[n]})
##        print(distanceval)
##        print(indexval)
##        print(Ytraindata[indexval],': this is the predicted cancer')
##        print(Ytestdata[n],': this is the actual cancer')
        if Ytraindata[indexval]==Ytestdata[n]:
            count=count+1
    print(count)
    print(count/140)
        
