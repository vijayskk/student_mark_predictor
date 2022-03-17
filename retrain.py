import tensorflow as tf
import keras
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib
data = pd.read_csv('student-por.csv',sep=";")


data = data[["G1","G2","G3","studytime","failures","absences"]]


wanted_predict = "G3"

X = np.array(data.drop([wanted_predict],axis=1))
Y = np.array(data["G3"])

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.1)


# plt.scatter(data["G1"],data["G3"],color='red')
# plt.scatter(data["G2"],data["G3"],color='blue')
# plt.scatter(data["studytime"],data["G3"],color='green')
# plt.plot(np.unique(data["G2"]), np.poly1d(np.polyfit(data["G2"], data["G3"], 1))(np.unique(data["G2"])),c='blue')
# plt.plot(np.unique(data["G1"]), np.poly1d(np.polyfit(data["G1"], data["G3"], 1))(np.unique(data["G1"])),c='red')
# plt.plot(np.unique(data["studytime"]), np.poly1d(np.polyfit(data["studytime"], data["G3"], 1))(np.unique(data["studytime"])),c='green')

# plt.plot()
# plt.show()



# print(Xtrain,"\n",Xtest,"\n",Ytrain,"\n",Ytest,"\n" )

model = linear_model.LinearRegression()

model.fit(Xtrain,Ytrain)

acc = model.score(Xtest,Ytest)
print(acc)

joblib.dump(model,'model.joblib')

