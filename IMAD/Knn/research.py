import pandas as pd
from Knn import Knn
from votes import majority_vote,distance_weighted_vote,farest_distance_weighted_vote
from Distances import euclidean,manhattan
from preprocess import preprocess_dateset

iris = pd.read_csv('datasets\Iris.csv')
iris = preprocess_dateset(iris,['Species'],['Id'])
glass = pd.read_csv('datasets\glass.csv')
glass = preprocess_dateset(glass,['Type'],['Id'])
wine = pd.read_csv('datasets\wine.csv')
wine = preprocess_dateset(wine,['quality'],[])
seeds = pd.read_csv('datasets\seeds.csv')
seeds = preprocess_dateset(seeds,['Type'],[])
#s = Knn(euclidean, majority_vote, 4)
# print(s.predicts(iris,iris,3))
#s.test(iris, 5)

def reseach(vote,distance,folds,K,title,subtitle):
    results_col = {'Set':[],
               'precision':[],
               'recall':[],
               'fscore':[]}
    results = pd.DataFrame(data=results_col)
    knn = Knn(distance,vote,K)
    print('---------IRIS ---------')
    results= results.append(knn.test(iris,folds,'Iris'+subtitle))
    print('---------GLASS ---------')
    knn = Knn(distance, vote, K)
    results = results.append(knn.test(glass,folds,'Glass'+subtitle))
    print('---------WINE ---------')
    knn = Knn(distance, vote, K)
    results = results.append(knn.test(wine, folds, 'Wine'+subtitle))
    print('---------Seeds ---------')
    knn = Knn(distance, vote, K)
    results = results.append(knn.test(seeds, folds, 'seeds '+subtitle))
    filename = 'Knn-'+title+str(k)+'folds'+str(fold)+'.csv'
    results.to_csv(filename)
    print(results.to_latex())
folds = [3,5,8]
K = [3, 5, 0]

for fold in folds:
    for k in K:
        subtitle = "K:"+str(k)+" Folds:"+str(fold)
        print('K:',k,' folds: ',fold)
        print('**Majority Vote')
        print('*******euclidean')
        reseach(majority_vote,euclidean,fold,k,'majorityEuc',subtitle+"majority "+"euclidean")
        print('*******manhattan')
        reseach(majority_vote,manhattan,fold,k,'majority',subtitle+"majority "+"manhattan")
        print('***distance_weighted_vote')
        print('*******euclidean')
        reseach(distance_weighted_vote,euclidean,fold,k,'distanceEuc',subtitle+"distance_weighted_vote "+"euclidean")
        print('*******manhattan')
        reseach(distance_weighted_vote,manhattan,fold,k,'distanceMan',subtitle+"distance_weighted_vote "+"manhattan")
        print('***farest_distance_weighted_vote')
        print('*******euclidean')
        reseach(farest_distance_weighted_vote,euclidean,fold,k,'FatestEuc',subtitle+"farest_distance_weighted_vote "+"euclidean")
        print('*******manhattan')
        reseach(farest_distance_weighted_vote,manhattan,fold,k,'FatestMan',subtitle+"farest_distance_weighted_vote "+"manhatan")