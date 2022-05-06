from preprocess import change_category_columns
from votes import majority_vote, distance_weighted_vote
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
from math import sqrt
class Knn:
    def __init__(self, distance_func, vote_func, K):
        self.distance_func = distance_func
        self.vote_func = vote_func
        self.X = None
        self.K = K
        pass

    def _get_neighbors(self, test_row):
        distances = []
        for index, train_row in self.X.iterrows():  # for each row in dataframe
            dist = self.distance_func(train_row[0:-1], test_row)  # train_row without class
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = []
        for i in range(self.K):
            neighbors.append(distances[i][0])

        return neighbors

    def _predict(self, row):
        neighbours = self._get_neighbors(row)
        prediction = self.vote_func(neighbours, self, row)
        return prediction

    def predicts(self, training_set, test_set):
        self.X = training_set
        predictions = []
        for i in range(len(test_set)):
            row = test_set.iloc[i, :]
            prediction = self._predict(row)
            predictions.append(prediction)
        return predictions

    def test(self, dataset, folds,title):

        raports = []
        skf = StratifiedKFold(n_splits=folds,shuffle=True)
        X = dataset.iloc[:, :]
        Y = dataset.iloc[:, -1]
        for train_index, test_index in skf.split(X, Y):
            train = X.iloc[train_index, :]
            test = X.iloc[test_index, 0:-1]
            if self.K == 0:
                self.K = int(sqrt(len(train)))
            classes_test = Y.iloc[test_index]
            prediciton = self.predicts(train, test)
            raports.append(classification_report(classes_test, prediciton, output_dict=True,zero_division=1))

        n_raports = len(raports)
        n_classes = len(raports[0].keys())-3
        div = n_raports*n_classes
        precision = 0
        recall = 0
        fscore = 0
        for raport in raports:
            for key in raport:
                if(key=='accuracy' or key=='macro avg'):
                    break
                precision +=raport[key]['precision']
                recall += raport[key]['recall']
                fscore += raport[key]['f1-score']
        print('precision',(precision/div),'recall',(recall/div),'fscore',(fscore/div))
        results_col = {'Set': [title],
                       'precision': [precision/div],
                       'recall': [recall/div],
                       'fscore': [fscore/div]}
        results = pd.DataFrame(data=results_col)
        return results




