from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
labels = ["employee", "manager", "director"]


def raport(model, train, test, y_train, y_test, plot_cf=False,raport_print=True):
    x = model.predict(train)


    train_raport = classification_report(y_train, x, target_names=labels,output_dict=True)
    if raport_print:
        print("***TRAIN SET RAPORT***")
        print(classification_report(y_train, x, target_names=labels))
    if plot_cf:
        plot_confusion_matrix(model, train, y_train)
        plt.title("Train set")
        plt.show()

    x = model.predict(test)
    test_raport = classification_report(y_test, x, target_names=labels, output_dict=True)
    if raport_print:
        print("***TEST SET RAPORT***")
        print(classification_report(y_test, x, target_names=labels))

    if plot_cf:
        plot_confusion_matrix(model, test, y_test)
        plt.title("Test set")
        plt.show()
    return train_raport,test_raport

def raport_fast(model, train, test, y_train, y_test):
    x = model.predict(train)
    train_raport = classification_report(y_train, x, output_dict=True)
    x = model.predict(test)
    test_raport = classification_report(y_test, x,  output_dict=True)
    return train_raport, test_raport