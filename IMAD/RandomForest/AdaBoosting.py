#Script to test AdaBoost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report

iris = pd.read_csv('datasets\Iris.csv')
#iris = preprocess_dateset(iris,['Species'],['Id'])

X = iris.iloc[:,1:-1] #Iris with no ID and Species columns
Y = iris.iloc[:,-1] # Iris Species columns (last one)
x_train , x_test, y_train,  y_test = train_test_split(X,Y,test_size=0.2)
print("AdaBoost")

ada_Boost = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10)
ada_Boost.fit(x_train,y_train)

print(classification_report(y_test, ada_Boost.predict(x_test), output_dict=False,zero_division=1))