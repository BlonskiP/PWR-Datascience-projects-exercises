from Knn import Knn
from votes import majority_vote,distance_weighted_vote,farest_distance_weighted_vote
from Distances import euclidean,manhattan
import pandas as pd
from preprocess import preprocess_dateset
import numpy as np
import matplotlib.pyplot as plt

# Loading some example data


dataset = pd.read_csv('datasets\Iris.csv')
dataset_preprcessed = preprocess_dateset(dataset,['Species'],['Id','PetalLengthCm','PetalWidthCm'])
X = dataset_preprcessed.iloc[:, 0:-1]
Y = dataset_preprcessed.iloc[:, -1]


#print(X)
#print(Y)
distance = manhattan
vote = majority_vote
K = 3


#predictions = knn.predicts(X,X)

print(dataset_preprcessed.iloc[:,1].min())

x_min, x_max = dataset_preprcessed.iloc[:,0].min() - 1, dataset_preprcessed.iloc[:,0].max() + 1
y_min, y_max = dataset_preprcessed.iloc[:,1].min() - 1, dataset_preprcessed.iloc[:,1].max() + 1


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


row = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
#print(X_plot)
#print(row)
knn = Knn(distance,vote,K)
pred = np.array(knn.predicts(dataset_preprcessed,row))
#print(xx.shape)
pred = pred.reshape(xx.shape)
print(pred)
fig, ax = plt.subplots()
ax.set_title('Iris with manhattan distance K=3')
ax.contourf(xx,yy,pred,alpha=0.8)
ax.scatter(dataset_preprcessed.iloc[:, 0], dataset_preprcessed.iloc[:, 1], c=Y,s=20, edgecolor='k')
plt.savefig('examplePlot.png')
plt.show()
