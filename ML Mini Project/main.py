import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Student_Performance.csv')
df.drop(['Extracurricular Activities'], axis=1, inplace=True)
X=df.drop('Performance Index',axis=1)
y=df['Performance Index']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
import pickle
pickle.dump(lr, open('model2.pkl', 'wb'))