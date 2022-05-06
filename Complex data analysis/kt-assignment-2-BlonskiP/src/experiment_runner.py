from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.svm import LinearSVC
import networkx as nx
import pandas as pd
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.metrics import multi_labeled, h_fbeta_score


def run_flat(clf_name,dataset):
    #Do some datashit
    y_train = dataset.records_train['labels'].apply(lambda x: x[-1])
    X_train = dataset.records_train.drop(columns=['labels'])

    y_test = dataset.records_test['labels'].apply(lambda x: x[-1])
    X_test = dataset.records_test.drop(columns=['labels'])
    #create cls
    clf = get_base(clf_name)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    df = metrics(y_pred,y_test,clf_name,dataset.hierarchy,"FLAT")
    #df = create_raport_dic(y_test,y_pred,clf_name)
    return df

def get_base(clf_name):
    clf=None
    clf_name = clf_name.upper()
    if clf_name == "RANDOM_FOREST":
        clf = RandomForestClassifier()
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=3)
    elif clf_name == "NN":
        clf = MLPClassifier(random_state=1, max_iter=100)
    return clf

def create_raport_dic(y_true,y_pred,clf_name):
    raport_dict = classification_report(y_true, y_pred, output_dict=True)
    keys = ['accuracy', 'macro avg', 'weighted avg']
    x = [pd.Series(raport_dict[d], name=d) for d in raport_dict if d not in keys]
    df = pd.DataFrame(x)
    df['clf'] = clf_name
    return df

def run_LCPN(clf_name,dataset):

    y_train = dataset.records_train['labels'].apply(lambda x: x[-1])
    X_train = dataset.records_train.drop(columns=['labels'])

    y_test = dataset.records_test['labels'].apply(lambda x: x[-1])
    X_test = dataset.records_test.drop(columns=['labels'])

    clf = HierarchicalClassifier(
        base_estimator=get_base(clf_name),
        class_hierarchy=dataset.hierarchy,
        algorithm='lcpn'
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    df = metrics(y_pred,y_test,clf_name,clf.graph_,"LCPN")

    return df

def metrics(y_pred,y_true,name,hierarchy,cls_type):
    with multi_labeled(y_true, y_pred,hierarchy) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_
        )
    #df = create_raport_dic(y_true, y_pred, name)
    df = pd.DataFrame({
        "h_fscore":[h_fbeta],
        "clf":[name],
        "cls_type":[cls_type]
    })
    return df

def run_LCN(clf_name,dataset):
    mlb = MultiLabelBinarizer()
    y_train = dataset.records_train['labels']
    X_train = dataset.records_train.drop(columns=['labels'])

    y_test = dataset.records_test['labels']
    X_test = dataset.records_test.drop(columns=['labels'])

    clf_lcn = OneVsRestClassifier(estimator=get_base(clf_name))
    clf_lcn.fit(X_train, mlb.fit_transform(y_train))

    y_pred = clf_lcn.predict(X_test)
    y_pred = mlb.inverse_transform(y_pred)
    #hierarchy=nx.from_dict_of_lists(hierarchy)
    df = metrics(y_pred,y_test,clf_name,dataset.hierarchy,"LCN")
    return df

def run_BigBang(clf_name,dataset):

    y_train = dataset.records_train['labels']
    X_train = dataset.records_train.drop(columns=['labels'])

    y_test = dataset.records_test['labels']
    X_test = dataset.records_test.drop(columns=['labels'])

    clf_big_bang = make_pipeline(
        TruncatedSVD(n_components=30),
        get_base(clf_name)
    )
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    #y= mlb.fit_transform(y_train)
    clf_big_bang.fit(X_train,y_train)
    #print(type(clf_big_bang))
    y_pred = clf_big_bang.predict(X_test)
    y_pred = mlb.inverse_transform(y_pred)
    #print(type(y_pred))
    #print(y_pred)
    df = metrics(y_pred, y_test, clf_name, dataset.hierarchy,"BIGBANG")
    return df

def run_LCPL(clf_name,dataset):

    y_train = dataset.records_train['labels']
    X_train = dataset.records_train.drop(columns=['labels'])

    y_test = dataset.records_test['labels']
    X_test = dataset.records_test.drop(columns=['labels'])

    levels = len(y_train[0])
    lens = y_train.apply(lambda x: len(x)==levels)
    assert lens.all()
    assert y_test.apply(lambda x: len(x)==levels).all()
    cls_per_level = []
    cls_pred = {}
    for lvl in range(levels):
        cls_per_level.append(get_base(clf_name))
        cls_per_level[lvl].fit(X_train, y_train.apply(lambda x: x[lvl]))

    for lvl in range(levels):
        cls_pred[lvl]=list(cls_per_level[lvl].predict(X_test))
    #print(cls_pred)
    y_pred = tuple(zip(*cls_pred.values()))
    #print(type(y_pred))
    #print(y_pred)
    df = metrics(y_pred, y_test, clf_name, dataset.hierarchy,"LCPL")
    return df







