import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report, make_scorer

from Knn.preprocess import preprocess_dateset
from RandomForest.utils import classification_report_score, create_reseach_result_dataframe, classification_reports
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
iris = pd.read_csv('datasets\Iris.csv')
iris = preprocess_dateset(iris,['Species'],['Id'])
glass = pd.read_csv('datasets\glass.csv')
glass = preprocess_dateset(glass,['Type'],['Id'])
wine = pd.read_csv('datasets\wine.csv')
wine = preprocess_dateset(wine,['quality'],[])
seeds = pd.read_csv('datasets\seeds.csv')
seeds = preprocess_dateset(seeds,['Type'],[])
max_depth_values = [5,10,None]
learning_rate = [1.0,0.9,0.85]
#base on previous research :
best_n_estimators = {
    "Wine" : 10,
    "Glass": 10,
    "Seeds": 30,
    "Iris" : 30
}
folds = [5, 8, 10]
results_col = {'Set': [],
               'precision': [],
               'recall': [],
               'fscore': []}
def xstr(s):
    if s is None:
        return 'None'
    return str(s)

def research_dataset(dataset,dataset_name):
    n_estimators = best_n_estimators[dataset_name]
    results = pd.DataFrame(data=results_col)
    str_param = dataset_name + " AdaBoost est=" + str(n_estimators)
    X = dataset.iloc[:, 0:-1]  # Iris with no ID and Species columns
    Y = dataset.iloc[:, -1]  # Iris Species columns (last one)
    for depth in max_depth_values:
        for rate in learning_rate:
            for fold in folds:
                bagging = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),n_estimators=n_estimators,learning_rate=rate)
                str_param_temp = str_param+" lr: "+xstr(rate) +" max_depth: "+xstr(depth) + " Fold:" + str(fold)
                cross_val_score(bagging, X=X, y=Y, cv=fold, scoring=make_scorer(classification_report_score))
                df = create_reseach_result_dataframe(classification_reports, str_param_temp)
                results = results.append(df)
    #with pd.option_context("max_colwidth", 1000):
    #    print(results.to_latex(index_names=False))
    return results
def put_to_file(research_result,dataset_name):
    filename = dataset_name+'.csv'
    print("***"+dataset_name+"***"+"\n")
    result = research_result.sort_values(by=['fscore'], ascending=False)
    with pd.option_context("max_colwidth", 1000):
        print(result.head(10).to_latex(index=False,caption="Najlepszy zestaw parametrów dla AdaBoost "+dataset_name))
    result.to_csv("Adaboost_BestParams_"+filename)
#put_to_file(research_dataset(iris,"Iris"),"Iris")
put_to_file(research_dataset(glass,"Glass"),"Glass")
put_to_file(research_dataset(wine,"Wine"),"Wine")
put_to_file(research_dataset(seeds,"Seeds"),"Seeds")

