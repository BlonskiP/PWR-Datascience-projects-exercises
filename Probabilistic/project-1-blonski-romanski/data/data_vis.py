import pandas as pd
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt

PADDING = 1

def cls_dist(data: pd.DataFrame, class_name: str = 'Churn',toFile=False):
    data[class_name].value_counts().sort_index().plot(kind='bar', title=('Dystrybucja: '+class_name),rot=1)
    if(toFile):
        plt.savefig('output-dist'+class_name+'.png')


def violine_plots(data: pd.DataFrame,
    labels: List[str] = ['tenure', 'MonthlyCharges', 'TotalCharges'],
    class_column: str = 'Churn',toFile=False):

    width = 2
    height = (len(labels)+1 // width) + 1
    label_number = 0
    
    f, axes = plt.subplots(height, 1, figsize=(20,20))
    for i in range(height):
        if label_number < len(labels):
            sns.violinplot(x=class_column, y=labels[label_number], data=data, ax=axes[i], title=labels[label_number], showmeans=True)
            label_number += 1
        else:
            plt.delaxes(axes[i])

    plt.tight_layout(pad=PADDING)
    if(toFile):
        f.savefig("output"+str(labels))
    else:
        display(plt.show())

def pearson_corr(data,toFile):
    sns.heatmap(data.corr())
    if toFile:
        plt.savefig("outputHeatMap.png")

def bar_plots(data: pd.DataFrame,
    labels: List[str] = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'],
    class_column: str = 'Churn',toFile=False):
    
    width = 2
    height = (len(labels)+1 // width) + 1
    label_number = 0
    
    f, axes = plt.subplots(height, width, figsize=(20,20))
    for i in range(height):
            if label_number < len(labels):
                data2 = data[class_column].groupby(data[labels[label_number]]).value_counts(normalize=True).rename(
                    'Proportion').reset_index()
                sns.countplot(x=class_column, hue=labels[label_number], data=data, ax=axes[i][0])
                sns.barplot(x=class_column, y='Proportion', hue=labels[label_number], data=data2, ax=axes[i][1])
                label_number += 1
            else:
                plt.delaxes(axes[i,0])
                plt.delaxes(axes[i, 1])
    
    plt.tight_layout(pad=PADDING)
    if toFile:
        f.savefig("output" + str(labels))
    else:
        display(plt.show())
