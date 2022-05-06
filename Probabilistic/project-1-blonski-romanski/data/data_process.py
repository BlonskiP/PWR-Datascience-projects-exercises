import pandas as pd
import os
BINARY_COLUMNS = ['Partner','Dependents','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
MULTIPLE_LINES_CATEGORIES = ''
def empty_totalcharges_to_zero(data: pd.DataFrame) -> pd.DataFrame:
    def get_val(row):
        if row['TotalCharges'] == ' ':
            row['TotalCharges'] = 0
        return row

    return data.apply(get_val, axis=1)

def set_types(data: pd.DataFrame, categorize=True) -> pd.DataFrame:
    f = ['tenure', 'MonthlyCharges', 'TotalCharges']#numeric
    for col in data.columns:
        if col in f:
            data[col] = data[col].astype('float64')
        else:
            if categorize:
                data[col] = pd.Categorical(data[col], categories=data[col].unique()).codes
            data[col] = data[col].astype('category')
    return data

def set_seniorcitizen_values(data: pd.DataFrame) -> pd.DataFrame:
    data['SeniorCitizen'] = data['SeniorCitizen'].map({1:'Yes', 0:'No'})
    return data

def create_processed_file(data,filename):
    new_data = empty_totalcharges_to_zero(data)
    new_data = set_types(new_data, True)
    new_filename = 'After_processed' + filename
    new_data.to_csv(new_filename)


data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/Telco-Customer-Churn.csv'))
data = empty_totalcharges_to_zero(data)