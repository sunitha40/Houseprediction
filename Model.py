import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("C:\\Users\\SRamayanam\\Downloads\\USA_Housing.csv")


df.drop(['Address'],axis =1, inplace = True)

X = df.drop(['Price'],axis = 1)
y = df['Price']

from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()

X = pd.DataFrame(pre_process.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7 ,test_size = 0.3, random_state=2)


from sklearn.linear_model import LinearRegression
Lr = LinearRegression()

Lr.fit(X_train, y_train)

pickle.dump(Lr,open("model.pkl","wb"))

model=pickle.load(open('model.pkl','rb'))