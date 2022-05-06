from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pyro
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from data.data_process import empty_totalcharges_to_zero, set_types
from argparse import Namespace
import torch
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns

def train_and_score_model(model, guide, X_train, X_test, y_true, epochs=500, lr=1e-1, test_samples=10, toFile=False, title=''):
    pyro.clear_param_store()

    history = train_svi(model, guide, X_train, num_epochs=epochs, lr=lr)
    visualize_loss(history)

    results = {
        'acc': [],
        '0_prec': [],
        '1_prec': [],
        '0_rec': [],
        '1_rec': [],
        '0_f1': [],
        '1_f1': []
    }

    all_y_true = []
    all_y_pred = []

    for i in range(test_samples):
        y_pred = model(X_test)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        res = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
        results['acc'].append(acc)
        results['0_prec'].append(res[0][0])
        results['1_prec'].append(res[0][1])
        results['0_rec'].append(res[1][0])
        results['1_rec'].append(res[1][1])
        results['0_f1'].append(res[2][0])
        results['1_f1'].append(res[2][1])
        all_y_true = all_y_true + list(y_true)
        all_y_pred = all_y_pred + list(y_pred)

    for key, value in results.items():
        results[key] = round(np.mean(np.array(results[key])), 3)

    print()
    print('Accuracy: ',results['acc'])
    df = pd.DataFrame([[results['0_prec'], results['0_rec'], results['0_f1']], [results['1_prec'], results['1_rec'], results['1_f1']]], columns=['Precision', 'Recall', 'F1Score'])
    print(df)
    display_confusion_matrix(all_y_true, all_y_pred,title=title,toFile=toFile)
    return df
    # return results

def display_confusion_matrix(y_true, y_pred, title='',toFile=False):
    
    classes_list = list(set(y_true))
    
    matrix = confusion_matrix(y_true, y_pred, classes_list)
    df_cm = pd.DataFrame(matrix, index = classes_list,
                      columns = classes_list)

    plt.figure(figsize=(6,5))
    plt.title("Macierz pomyłek "+str(title))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.yticks(rotation=0)
    plt.ylabel('Wartości prawdziwe')
    plt.xlabel('Wartości predykowane')
    
    a, b = plt.ylim()
    a += 0.5
    b -= 0.5
    plt.ylim(a, b)
    if toFile:
        plt.savefig('MacierzPomyłek'+title+".png")
    else:
        display(plt.show())

def train_svi(model, guide, X, num_epochs=500, lr=1e-2):
    pyro.clear_param_store()

    svi = pyro.infer.SVI(
        model=model,
        guide=guide,
        optim=pyro.optim.Adam({'lr': lr}),
        loss=pyro.infer.TraceEnum_ELBO(),
    )

    history = {
        'losses': [],
        'params': defaultdict(list),
    }

    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            loss = svi.step(X)
            history['losses'].append(loss)

            if epoch % 50 == 0:
                print(f'Loss = {loss}')

            p = dict(pyro.get_param_store())
            for k, v in p.items():
                history['params'][k].append(v.detach().numpy().copy())

    return history

def visualize_loss(history):
    num_epochs = len(history['losses'])
    losses = history['losses']
    params = history['params']

    plt.figure(figsize=(15, 5))
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')

def get_processed_data(csv_path, balanced=False, seed=None, disc_bins=3):

    data = pd.read_csv(csv_path)
    data = empty_totalcharges_to_zero(data)
    data = set_types(data, categorize=True)

    if balanced:
        data=data.drop(data.sample(frac=1, random_state=seed)[data['Churn']==0].head(3305).index)

    disc = KBinsDiscretizer(disc_bins, encode='ordinal')

    data['tenure'] = disc.fit_transform(data['tenure'].values.reshape(-1, 1))
    data['MonthlyCharges'] = disc.fit_transform(data['MonthlyCharges'].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(data.drop('Churn', axis=1), data['Churn'], test_size=0.2, random_state=seed)

    x_y_train = Namespace(
        SeniorCitizen = torch.tensor(X_train['SeniorCitizen'].values).float(),
        Partner = torch.tensor(X_train['Partner'].values).float(),
        InternetService = torch.tensor(X_train['InternetService'].values).float(),
        DeviceProtection = torch.tensor(X_train['DeviceProtection'].values).float(),
        OnlineBackup = torch.tensor(X_train['OnlineBackup'].values).float(),
        Dependents = torch.tensor(X_train['Dependents'].values).float(),
        OnlineSecurity = torch.tensor(X_train['OnlineSecurity'].values).float(),
        TechSupport = torch.tensor(X_train['TechSupport'].values).float(),
        Contract = torch.tensor(X_train['Contract'].values).float(),
        PaperlessBilling = torch.tensor(X_train['PaperlessBilling'].values).float(),
        PaymentMethod = torch.tensor(X_train['PaperlessBilling'].values).float(),
        Tenure = torch.tensor(X_train['tenure'].values).float(),
        MonthlyCharges = torch.tensor(X_train['MonthlyCharges'].values).float(),
        y = torch.tensor(y_train.values).float(),

        shape=X_train.shape
    )

    x_test = Namespace(
        SeniorCitizen = torch.tensor(X_test['SeniorCitizen'].values).float(),
        Partner = torch.tensor(X_test['Partner'].values).float(),
        InternetService = torch.tensor(X_test['InternetService'].values).float(),
        DeviceProtection = torch.tensor(X_test['DeviceProtection'].values).float(),
        OnlineBackup = torch.tensor(X_test['OnlineBackup'].values).float(),
        Dependents = torch.tensor(X_test['Dependents'].values).float(),
        OnlineSecurity = torch.tensor(X_test['OnlineSecurity'].values).float(),
        TechSupport = torch.tensor(X_test['TechSupport'].values).float(),
        Contract = torch.tensor(X_test['Contract'].values).float(),
        PaperlessBilling = torch.tensor(X_test['PaperlessBilling'].values).float(),
        PaymentMethod = torch.tensor(X_test['PaymentMethod'].values).float(),
        Tenure = torch.tensor(X_test['tenure'].values).float(),
        MonthlyCharges = torch.tensor(X_test['MonthlyCharges'].values).float(),
        y = None,

        shape=X_test.shape
    )

    return x_y_train, x_test, y_test
