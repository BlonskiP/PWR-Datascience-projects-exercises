import numpy as np
import pandas as pd

from model.embedder import Embedder
from model.dataset import purify_text

"""
Cosine distance symptoms classifier model 
"""


COL_NAME = "nazwa"
COL_SYMPTOMS = "objawy"

class CosineClassifier:

    def __init__(self,
                 embedding_model_path="dr_herbert_complete",
                 symptoms_path="objawy.csv",
                 cuda_enabled=True
                 ):

        self.embedder = Embedder(embedding_model_path,
                                 cuda=cuda_enabled,
                                 batch=4,
                                 word_embeddings=False)

        self.symptoms_df = pd.read_csv(symptoms_path, header=None, names=[COL_NAME, 'x', COL_SYMPTOMS])
        self.symptoms_df.dropna(inplace=True)
        symptoms_arr = self.symptoms_df[COL_SYMPTOMS].values.tolist()
        # print("Problematyczny: ", symptoms_arr[6])
        # texts = [purify_text(t) for t in symptoms_arr[:6] + symptoms_arr[7:]]
        texts = [purify_text(t) for t in symptoms_arr]
        self.symptoms_embds = self.embedder.make_embeddings(texts)

    def name(self):
        return self.model.name

    def predict(self, query):
        if type(query) == str:
            query = [query]

        query_arr = [purify_text(t) for t in query]
        query_embds = self.embedder.make_embeddings(query_arr)

        results = []

        for query_idx in range(len(query_embds)):
            x = query_embds[query_idx]
            y = self.symptoms_embds
            symptoms_distance = np.dot(y, x) / (np.linalg.norm(x) * np.linalg.norm(self.symptoms_embds, axis=-1))

            idx = np.argmax(symptoms_distance)
            diagnosis = self.symptoms_df[COL_NAME].values[idx]
            # print(query)
            # print(idx, diagnosis, symptoms_distance[idx])
            # for pair in zip(self.symptoms_df[COL_NAME].values, symptoms_distance):
            #     print(pair)

            results.append({
                'query' : query[query_idx],
                'diagnosis' : diagnosis,
                'distance' : symptoms_distance[idx]
            })

        return results
