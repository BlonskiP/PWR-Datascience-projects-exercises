import os
import glob
import pathlib
import pandas as pd
path = pathlib.Path().absolute()
extension = 'csv'
os.chdir(path)
csvs = glob.glob('*.{}'.format(extension))
results_col = {'Set':[],
               'precision':[],
               'recall':[],
               'fscore':[]}
top_results = pd.DataFrame(data=results_col)
for file in csvs:
    result = pd.read_csv(file)
    result = result.drop('Unnamed: 0', axis=1)
    top_results = pd.concat([top_results, result], axis=0, sort=False)
Iris = pd.DataFrame(data=results_col)
Glass = pd.DataFrame(data=results_col)
Wine = pd.DataFrame(data=results_col)
seeds = pd.DataFrame(data=results_col)
for index, row in top_results.iterrows():
    if row['Set'].startswith('Iris'):
        Iris=Iris.append(row)
    if row['Set'].startswith('Glass'):
        Glass = Glass.append(row)
    if row['Set'].startswith('Wine'):
        Wine=Wine.append(row)
    if row['Set'].startswith('seed'):
        seeds=seeds.append(row)
Iris.drop_duplicates(subset=['Set'],keep='first',inplace=True)
Glass.drop_duplicates(subset=['Set'],keep='first',inplace=True)
Wine.drop_duplicates(subset=['Set'],keep='first',inplace=True)
seeds.drop_duplicates(subset=['Set'],keep='first',inplace=True)

Iris= Iris.sort_values(by=['fscore'],ascending=False)
Glass = Glass.sort_values(by=['fscore'],ascending=False)
Wine = Wine.sort_values(by=['fscore'],ascending=False)
seeds= seeds.sort_values(by=['fscore'],ascending=False)

filename = 'Iris_res.csv'
Iris.to_csv(filename)

filename = 'Glass_res.csv'
Glass.to_csv(filename)

filename = 'Wine_res.csv'
Wine.to_csv(filename)

filename = 'seeds_res.csv'
seeds.to_csv(filename)
with pd.option_context("max_colwidth", 1000):
    print(Iris.head(10).to_latex(index_names=False))
    print(Glass.head(10).to_latex(index_names=False))
    print(Wine.head(10).to_latex(index_names=False))
    print(seeds.head(10).to_latex(index_names=False))