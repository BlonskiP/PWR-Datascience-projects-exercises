#Script to test bagging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report

iris = pd.read_csv('datasets\Iris.csv')
#iris = preprocess_dateset(iris,['Species'],['Id'])

X = iris.iloc[:,1:-1] #Iris with no ID and Species columns
Y = iris.iloc[:,-1] # Iris Species columns (last one)
x_train , x_test, y_train,  y_test = train_test_split(X,Y,test_size=0.2)
print("Bagging")

baggingClassifier = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10)
baggingClassifier.fit(x_train,y_train)

print(classification_report(y_test, baggingClassifier.predict(x_test), output_dict=False,zero_division=1))