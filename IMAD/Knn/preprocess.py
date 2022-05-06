from sklearn.preprocessing import StandardScaler
import pandas as pd

def drop_columns(dataset,columns_to_drop):
    dataset = dataset.drop(columns_to_drop,axis=1)
    return dataset

def change_category_columns(dataset, cat_col):
    dic = {}
    for col in cat_col:
        dic[col] = 'category'
    new_set = dataset.astype(dic)
    cat_columns = new_set.select_dtypes(['category']).columns
    new_set[cat_columns] = new_set[cat_columns].apply(lambda x: x.cat.codes)
    new_set = new_set.astype(dic)
    return new_set

def preprocess_dateset(dataset,cat_col=None,drop_col=None):
    dataset = drop_columns(dataset,drop_col)
    dataset = change_category_columns(dataset,cat_col)
    dataset = StandarScalce(dataset)
    dataset = change_category_columns(dataset,cat_col)
    return dataset
def StandarScalce(dataset):
    cat_columns = dataset.select_dtypes(include=['category'])
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset.select_dtypes(exclude=['category']).values)
    dataset=pd.DataFrame(dataset)
    dataset = pd.concat([dataset, cat_columns], axis=1, sort=False)
    return dataset.astype('float64',copy=True)