import matplotlib.pyplot as plt
import pyro
from sklearn.model_selection import KFold
category_col = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
from sklearn import metrics as sk_mtr


def show_summary(model, y_train, x_train, y_test, x_test):
    print(sk_mtr.classification_report(
        y_true=y_train,
        y_pred=model.predict(X=x_train), ))
    print(sk_mtr.classification_report(
        y_true=y_test,
        y_pred=model.predict(X=x_test), ))

def plot_params(history,toFile=False):
    #Ploting numeric (gausian)

    num_epochs = len(history['losses'])
    losses = history['losses']
    params = history['params']
    xs = range(num_epochs)
    plt.figure(figsize=(15, 5))
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    Numeric_params_names = []
    for name in numeric_cols:
        Numeric_params_names.append(name+'_mu')
        Numeric_params_names.append(name + '_sigma')
    for pname, pvals in params.items():
        if pname in Numeric_params_names:
            plt.figure(figsize=(15, 5))
            plt.plot(xs, pvals, label=f'P({pname} )')
            plt.title(pname)
        else:
            plt.figure(figsize=(15, 5))
            for i in (0, 1):
                vals = [v[i] for v in pvals]
                plt.plot(xs, vals, label=f'P({pname} class:{i})')
            plt.title(pname)
        plt.legend()
        diff = pvals[-1] - pvals[0]
        if toFile:
            plt.savefig('PyroParams'+pname+'.png')
        print(f'{pname} Diffrence betweend start and end param value', diff)

def test_model(model, folds, data):
    kf = KFold(n_splits=folds)
    results = []
    for train_index, test_index in kf.split(data):
        train_x = data.iloc[train_index, 1:20]
        train_y = data.iloc[train_index, -1]
        model.fit(data_X=train_x, data_Y=train_y)
        test_x = data.iloc[test_index, 1:20]
        test_y = data.iloc[test_index, -1]
        results.append(model.test(test_x, test_y))
    avg_fscore_0 = 0
    avg_fscore_1 = 0
    avg_prec_0 = 0
    avg_prec_1 = 0
    avg_rec_0 = 0
    avg_rec_1 =0
    avg_acc = 0
    for res in results:
        avg_fscore_0 += res['0']['f1-score'] / len(results)
        avg_fscore_1 += res['1']['f1-score'] / len(results)

        avg_prec_0 += res['0']['precision'] / len(results)
        avg_prec_1 += res['1']['precision'] / len(results)

        avg_rec_0 += res['0']['recall'] / len(results)
        avg_rec_1 += res['1']['recall'] / len(results)

        avg_acc += res['accuracy'] / len(results)

    res_dic = {
        'avg_fscore':[round(avg_fscore_0,4),round(avg_fscore_1,4)],
        'avg_precision':[round(avg_prec_0,4),round(avg_prec_1,4)],
        'avg_recall': [round(avg_rec_0, 4), round(avg_rec_1, 4)],
        'avg_accuracy':round(avg_acc, 4)
    }
    return res_dic
