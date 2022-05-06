import pandas as pd
import sys
sys.path.append('..')
from data.data_process import create_processed_file, set_types,empty_totalcharges_to_zero
from src import utils
from data.data_vis import cls_dist,violine_plots,pearson_corr , bar_plots
from src.BayesianNetwork.utils import get_processed_data, train_and_score_model
from src.BayesianNetwork.models import model_1, model_2, model_3_1, model_3_2, model_3_3, model_3_4, model_3_5, model_3_6, model_4_1, model_4_2, model_4_3
from src.NB.NB_Prototype import NaiveBayes as nb
DATASET_FILENAME = 'Telco-Customer-Churn.csv'
AFTER_PROC_DATASET_FILENAME = 'After_processedTelco-Customer-Churn.csv'
#Check if have data file
def nb_research(data, results_file):

    def NB_test(data,results_file,folds):
        results_file.write('Folds' + str(folds) + '\n')
        lr = 1e-2  # learning rate
        NB = nb(lr, 100)
        res = utils.test_model(NB, folds, data)
        results_file.write(str(res)+'\n')
        return NB
    print(data.info())
    results_file.write('********Naive Bayes Research******** \n')
    results_file.write('Data len:'+str(len(data))+'\n')
    results_file.write('Fscore NO  Fscore YES \n')
    last_nb = NB_test(test_data, results_file, 3)
    utils.plot_params(last_nb.history, True)
    NB_test(test_data, results_file, 5)
    NB_test(test_data, results_file, 10)
def data_anal(data,results_file):
    results_file.write("DATASET HEAD \n")
    results_file.write(str(data.head(5)))
    results_file.write('\n')


    cls_dist(data,toFile=True)
    cls_dist(data, 'InternetService', toFile=True)
    cls_dist(data, 'TechSupport', toFile=True)
    violine_plots(data,toFile=True)
    pearson_corr(data[['tenure', 'MonthlyCharges', 'TotalCharges']],toFile=True)
    bar_plots(data, labels=['gender', 'SeniorCitizen'],toFile=True)
    bar_plots(data, labels=['Partner', 'Dependents'],toFile=True)
    bar_plots(data, labels=['PhoneService', 'MultipleLines'],toFile=True)
    bar_plots(data, labels=['InternetService', 'OnlineSecurity'],toFile=True)
    bar_plots(data, labels=['OnlineBackup', 'DeviceProtection'],toFile=True)
    bar_plots(data, labels=['TechSupport', 'StreamingTV', 'StreamingMovies'],toFile=True)
    violine_plots(data,
                  labels=['tenure', 'MonthlyCharges', 'TotalCharges'],
                  class_column='InternetService',toFile=True)

    bar_plots(data, labels=['Churn', 'OnlineSecurity'], class_column='InternetService',toFile=True)
    bar_plots(data, labels=['OnlineBackup', 'DeviceProtection'], class_column='InternetService',toFile=True)
    bar_plots(data, labels=['TechSupport', 'StreamingTV'], class_column='InternetService',toFile=True)
    bar_plots(data, labels=['StreamingMovies'], class_column='InternetService',toFile=True)

    bar_plots(data, labels=['Contract', 'PaperlessBilling', 'PaymentMethod'],toFile=True)
    violine_plots(data,
                  labels=['tenure', 'MonthlyCharges', 'TotalCharges'],
                  class_column='Contract',toFile=True)
    violine_plots(data,
                  labels=['tenure', 'MonthlyCharges', 'TotalCharges'],
                  class_column='PaperlessBilling',toFile=True)
    violine_plots(data,
                  labels=['tenure', 'MonthlyCharges', 'TotalCharges'],
                  class_column='PaymentMethod',toFile=True)
    bar_plots(data, labels=['gender', 'SeniorCitizen', 'Partner', 'Dependents'], class_column='Contract',toFile=True)


def bayes_networks_research(data, results_file):
    x_y_train, x_test, y_test = get_processed_data(csv_path='Telco-Customer-Churn.csv', balanced=False,seed=997, disc_bins=3)

    def write_result_to_file(res, result_filem, model_str):
        results_file.write('*****BAYES NETWORK RESEARCH*********'+'\n')
        results_file.write('*****MODEL :'+model_str+'\n')
        results_file.write('Precision \n')
        results_file.write('0\t1 \n')
        results_file.write(str(res['Precision'][0])+"\t")
        results_file.write(str(res['Precision'][1])+'\n')
        results_file.write('F1Score \n')
        results_file.write('0\t1 \n')
        results_file.write(str(res['F1Score'][0]) + "\t")
        results_file.write(str(res['F1Score'][1]) + "\n")
        results_file.write('Recall \n')
        results_file.write('0\t1 \n')
        results_file.write(str(res['Recall'][0])+ '\t')
        results_file.write(str(res['Recall'][1])+ "\n")
        results_file.write("\n")

    def guide(x):
        pass
    model_str = 'model_'
    i = 0
    models = [model_1, model_2, model_3_1, model_3_2, model_3_3, model_3_4, model_3_5, model_3_6, model_4_1, model_4_2, model_4_3]
    for model in models:
        i += 1
        print()
        print('BayesianNetworks - model '+str(i)+' ...')
        print()
        title= (model_str+str(i))
        res = train_and_score_model(model, guide, x_y_train, x_test, y_test, test_samples=100,toFile=True,title=title)
        write_result_to_file(res,results_file,title)



try:
    #preprocessing
    print()
    print('Przygotowanie danych...')
    print()
    data = pd.read_csv(DATASET_FILENAME)
    test_data = empty_totalcharges_to_zero(data)
    test_data = set_types(test_data, False)
    results_file = open("results_file.txt", "a")
    #DataAnal
    print()
    print('Wizualizacja danych...')
    print()
    data_anal(test_data,results_file)
    test_data = set_types(test_data, True)
    print()
    print('NaiveBayes...')
    print()
    nb_research(data, results_file)
    print()
    print('BayesianNetworks...')
    print()
    bayes_networks_research(data,results_file)
    results_file.close()

except IOError:
    print("IO Error")
