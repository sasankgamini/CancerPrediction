import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import preprocessing, neighbors




df=pd.read_csv("breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
##print(df.head())
df.drop('id',1)
x=df.drop('class',1)
y=df['class']
print(y)




##x=preprocessing.scale(x)
####print(y)
##x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
##clf=neighbors.KNeighborsClassifier()
##clf.fit(x_train,y_train)
##accuracy=clf.score(x_test,y_test)
##print(accuracy)
####exampledata=np.array([1016277,6,8,8,1,3,4,3,7,1])
####exampledata=exampledata.reshape(1,-1)
####prediction=clf.predict(exampledata)
####print(prediction)
