import os
import gzip
import sys

import pandas as pd
import json

from src.preprocess import Preprocessor
import urllib.request
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class Data_loader:
    DATA_DIR_NAME = "COVID DATA"
    FILE_LIMIT = 15 #30 files is about 1 month
   # FILE_LIMIT = sys.maxsize
    FILES_URL = "https://zenodo.org/record/4053573#.X373K2gzZPY"
    DOMAIN_URL = "https://zenodo.org"

    def check_files_amount(self):
        files_count = 0
        for filename in os.listdir(Data_loader.DATA_DIR_NAME):
            if filename.lower().endswith('.gz'):
                files_count +=1
        return files_count

    def load_data(self):
        # Create new dir
        self.create_data_dir()
        #Download data
        if self.check_files_amount() < self.FILE_LIMIT:
            self.download_data()
        #load and return
        data = self.load_files()
        preprocessor = Preprocessor()
        summary = preprocessor.preprare_summary(data)
        print('loading done')
        return summary

    def create_data_dir(self):
        if not os.path.exists(Data_loader.DATA_DIR_NAME):
            os.mkdir(Data_loader.DATA_DIR_NAME)



    def download_data(self):
        with urllib.request.urlopen(self.FILES_URL) as response:
            html = response.read()
            soup = BeautifulSoup(html)
            links = []
            files_downloaded = 0
            for link in soup.findAll("a", {"class": "filename"}):
                # print(link.get('href'))
                links.append(link.get('href'))
            for link in links:
                if self.check_files_amount() >= self.FILE_LIMIT:
                    break
                print(link)
                a = urlparse(link)
                filename = os.path.basename(a.path)
                path = os.path.join(Data_loader.DATA_DIR_NAME, filename)
                if not os.path.isfile(path):
                    r = requests.get(self.DOMAIN_URL + link, allow_redirects=True)
                    open(path, 'wb').write(r.content)

    def load_files(self):
        preprocessor = Preprocessor()
        end_data = None
        files_loaded = 0
        for filename in os.listdir(Data_loader.DATA_DIR_NAME):
            if filename.lower().endswith('.gz') and files_loaded < self.FILE_LIMIT:
                path = os.path.join(Data_loader.DATA_DIR_NAME,filename)
                print(filename)
                with gzip.GzipFile(path, 'r', ) as fin:
                    data_lan = []
                    for line in fin:
                        data_lan.append(json.loads(line.decode('utf-8')))

                data = pd.DataFrame(data_lan)
                #df = pd.read_json(path,compression='gzip',lines=True)
                df = preprocessor.preprocess(data)
                if end_data is None:
                    end_data = df
                else:
                    end_data = end_data.append(df,ignore_index = True)
                files_loaded+=1
        return preprocessor.group(end_data)


