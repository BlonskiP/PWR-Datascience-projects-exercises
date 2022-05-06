import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer

classification_reports = []

def create_reseach_result_dataframe(classification_reports, str_params):
    n_raports = len(classification_reports)
    n_classes = len(classification_reports[0].keys()) - 3
    div = n_raports * n_classes
    precision = 0
    recall = 0
    fscore = 0
    for raport in classification_reports:
        for key in raport:
            if (key == 'accuracy' or key == 'macro avg'):
                break
            precision += raport[key]['precision']
            recall += raport[key]['recall']
            fscore += raport[key]['f1-score']
   # print('precision', (precision / div), 'recall', (recall / div), 'fscore', (fscore / div))
    results_col = {'Set': [str_params],
                   'precision': [precision / div],
                   'recall': [recall / div],
                   'fscore': [fscore / div]}
    results = pd.DataFrame(data=results_col)
    return results

def classification_report_score(y_true, y_pred):

    raport = classification_report(y_true, y_pred, output_dict=True,zero_division=1)
    classification_reports.append(raport)
    return accuracy_score(y_true, y_pred)
