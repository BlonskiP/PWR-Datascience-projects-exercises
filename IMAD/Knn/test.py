import pandas as pd
from Knn import Knn
from votes import majority_vote,distance_weighted_vote,farest_distance_weighted_vote
from Distances import euclidean,manhattan
from preprocess import preprocess_dateset
K=3
Fold=5
glass = pd.read_csv('datasets\glass.csv')
glass = preprocess_dateset(glass,['Type'],['Id'])
knn = Knn(distance_func=manhattan,vote_func=distance_weighted_vote,K=K)
for i in range(0,5):
    knn.test(glass,5,"GLASS")