from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from RandomForest.utils import create_reseach_result_dataframe, classification_reports,classification_report_score
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from Knn.preprocess import preprocess_dateset
iris = pd.read_csv('datasets\Iris.csv')
iris = preprocess_dateset(iris,['Species'],['Id'])
glass = pd.read_csv('datasets\glass.csv')
glass = preprocess_dateset(glass,['Type'],['Id'])
wine = pd.read_csv('datasets\wine.csv')
wine = preprocess_dateset(wine,['quality'],[])
seeds = pd.read_csv('datasets\seeds.csv')
seeds = preprocess_dateset(seeds,['Type'],[])
results_col = {'Set': [],
                   'precision': [],
                   'recall': [],
                   'fscore': []}
folds = [5, 8, 10]
n_estimators = [10,20,30]
def research_Random_forest(n_estimators, dataset_name, dataset):
    results = pd.DataFrame(data=results_col)
    str_param = dataset_name + " RF estymatory="+str(n_estimators)
    X = dataset.iloc[:, 0:-1]  # Iris with no ID and Species columns
    Y = dataset.iloc[:, -1]  # Iris Species columns (last one)
    df = None
    for fold in folds:
        randomForest = RandomForestClassifier(n_estimators=n_estimators,  # how many decision trees inside?
                                              n_jobs=-1,  # use all processors for pararell operations
                                              )
        str_param_temp = str_param + " Fold:"+str(fold)
        cross_val_score(randomForest, X=X, y=Y, cv=fold, scoring=make_scorer(classification_report_score))
        df = create_reseach_result_dataframe(classification_reports, str_param_temp)
        results= results.append(df)

    return results

def research_Bagging(n_estimators, dataset_name, dataset):
    results = pd.DataFrame(data=results_col)
    str_param = dataset_name + " Bagging estymatory="+str(n_estimators)
    X = dataset.iloc[:, 0:-1]
    Y = dataset.iloc[:, -1]
    for fold in folds:
        bagging = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=n_estimators)
        str_param_temp = str_param + " Fold:"+str(fold)
        cross_val_score(bagging, X=X, y=Y, cv=fold, scoring=make_scorer(classification_report_score))
        df = create_reseach_result_dataframe(classification_reports, str_param_temp)
        results= results.append(df)
    return results

def research_AddaBoost(n_estimators, dataset_name, dataset):
    results = pd.DataFrame(data=results_col)
    str_param = dataset_name + " AdaBoost estymator="+str(n_estimators)
    X = dataset.iloc[:, 0:-1]
    Y = dataset.iloc[:, -1]
    for fold in folds:
        ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=n_estimators)
        str_param_temp = str_param + " Fold:"+str(fold)
        cross_val_score(ada, X=X, y=Y, cv=fold, scoring=make_scorer(classification_report_score))
        df = create_reseach_result_dataframe(classification_reports, str_param_temp)
        results= results.append(df)
    return results

def research_dataset(dataset_name,dataset):
    results_bagging = pd.DataFrame(data=results_col)
    results_ada = pd.DataFrame(data=results_col)
    results_randomForest = pd.DataFrame(data=results_col)
    for n_estimators_item in n_estimators:
        df_forest = research_Random_forest(n_estimators_item,dataset_name,dataset)
        results_randomForest= results_randomForest.append(df_forest)
        df_bagging = research_Bagging(n_estimators_item,dataset_name,dataset)
        results_bagging = results_bagging.append(df_bagging)
        df_adaboost = research_AddaBoost(n_estimators_item,dataset_name,dataset)
        results_ada = results_ada.append(df_adaboost)
    #return sorted by Fscore
    results_randomForest = results_randomForest.sort_values(by=['fscore'], ascending=False)
    results_bagging = results_bagging.sort_values(by=['fscore'], ascending=False)
    results_ada = results_ada.sort_values(by=['fscore'], ascending=False)
    return results_randomForest, results_bagging, results_ada

def put_to_file(research_results,dataset_name):
    filename = dataset_name+'.csv'
    research_results[0].to_csv("RF_"+filename, index=False)
    research_results[1].to_csv("Bagging_" + filename, index=False)
    research_results[2].to_csv("adaBoost_" + filename, index=False)
    print("***"+dataset_name+"***"+"\n")
    print(research_results[0].head(5).to_latex(index=False,caption="Najlepsze ilości estymatorów dla Random forest "+dataset_name))
    print(research_results[1].head(5).to_latex(index=False,
                                               caption="Najlepsze ilości estymatorów dla Bagging " + dataset_name))
    print(research_results[2].head(5).to_latex(index=False,
                                               caption="Najlepsze ilości estymatorów dla Adaboost " + dataset_name))
#put_to_file(research_dataset('iris',iris),"iris")
put_to_file(research_dataset('Glass',glass),"Glass")
put_to_file(research_dataset('wine',wine),"Wine")
put_to_file(research_dataset('seeds',seeds),"seeds")
