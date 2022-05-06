import pandas as pd
import os

DATA_PATH = f".\dane"
SERIES_FILES = ['series1.csv', 'series2.csv', 'series3.csv']


def load_all_series():
    dfs = []
    for idx, file in enumerate(SERIES_FILES):
        filepath = os.path.join(DATA_PATH, file)
        assert os.path.exists(filepath)
        df = pd.read_csv(filepath).rename(columns={'x': f'x_series_{idx}'})
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return df

def generator(series_number): # 1 2 3 as series number
    assert int(series_number) in [1,2,3]
    filename = SERIES_FILES[series_number-1]
    print(filename)
    filepath = os.path.join(DATA_PATH,str(filename))
    assert os.path.exists(filepath)
    with open(filepath, 'r') as f:
        for line in f:
            if 'x' in line: #first line has column name "x".
                pass
            else:
                yield float(line)
        yield None