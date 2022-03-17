import tensorflow as tf
import keras
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib import style


data = pd.read_csv('student-por.csv',sep=";")


data = data[["G1","G2","G3","studytime","failures","absences"]]


wanted_predict = "G3"

X = np.array(data.drop([wanted_predict],axis=1))
Y = np.array(data["G3"])

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.1)





# print(Xtrain,"\n",Xtest,"\n",Ytrain,"\n",Ytest,"\n" )

model = linear_model.LinearRegression()

model.fit(Xtrain,Ytrain)

acc = model.score(Xtest,Ytest)
print(acc)

joblib.dump(model,'model.joblib')

style.use("ggplot")

plt.scatter(data["G1"],data["G3"],color='red')
plt.xlabel("G1")
plt.plot()
plt.show()
plt.scatter(data["G2"],data["G3"],color='blue')
plt.xlabel("G2")
plt.plot()
plt.show()
