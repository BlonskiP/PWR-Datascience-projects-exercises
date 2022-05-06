#!/usr/bin/env python3

from threading import Lock
import os
import argparse

import tensorflow as tf
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np

from model.embedder import Embedder
from model.classifier import Classifier
from model.dataset import purify_text
from logger import get_logger

logger = get_logger("ml")
logger.info("ML logger init")

ENV_EMBEDDING_MODEL_PATH = "ML_EMBEDDING_MODEL_PATH"
ENV_CLF_MODEL_PATH = "ML_CLF_MODEL_PATH"
ENV_CUDA_ENABLED = "ML_CUDA"

class MLService:
    _instance = None
    _lock = Lock()

    def instance():
        with MLService._lock:
            if MLService._instance is None:
                MLService._instance = MLService()
            return MLService._instance

    def __init__(self, gpu_batch=4):
        if ENV_EMBEDDING_MODEL_PATH not in os.environ:
            logger.error(f"{ENV_EMBEDDING_MODEL_PATH} not set. Failing...")
            raise ValueError("Env")
        if ENV_CLF_MODEL_PATH not in os.environ:
            logger.error(f"{ENV_CLF_MODEL_PATH} not set. Failing...")
            raise ValueError("Env")
        cuda_enabled = ENV_CUDA_ENABLED in os.environ
        self.embedder = Embedder(os.environ[ENV_EMBEDDING_MODEL_PATH], cuda=cuda_enabled, batch=gpu_batch, word_embeddings=True)
        with tf.device('/CPU:0'):
            self.classifier = Classifier(os.environ[ENV_CLF_MODEL_PATH])

    def forward_pass(self, texts):
        logger.info(f"Purifing {len(texts)} sacret texts...")
        texts = [purify_text(t) for t in texts]
        logger.info(f"Embeddings for {len(texts)} sacret texts...")
        embeddings = self.embedder.make_embeddings(texts)
        logger.info(f"Classification for {len(texts)} sacret texts...")

        with tf.device('/CPU:0'):
            predictions = self.classifier.predict(embeddings)
            label_indices = np.argmax(predictions, axis=1)

        return label_indices.tolist()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Classify tweets on remote mongodb")
    parser.add_argument('mongourl', help="URL to mongo instance")
    parser.add_argument('--gpu-batch', dest='gpu_batch', type=int, default=32, help="GPU batch size")
    parser.add_argument('--save-batch', dest='save_batch', type=int, default=1000, help="GPU batch size")
    args = parser.parse_args()

    mongo_url = args.mongourl
    gpu_batch = args.gpu_batch
    save_batch = args.save_batch

    mongo = MongoClient(mongo_url)
    collection = mongo.database.tweets
    service = MLService(gpu_batch=gpu_batch)
    query = {
        "class": None,
        "lang": "pl",
        "datetime": { "$gte": "2020-10-01T00:00:00Z" } 
    }

    total_documents = collection.count_documents(query)
    pbar = tqdm(total=total_documents)
    while True:
        tweets = list(collection.find(query, limit=save_batch))
        if len(tweets) == 0:
            logger.info("No more tweets to classify")
            break

        texts = [t["tweet"] for t in tweets]
        ids = [t["_id"] for t in tweets]
        predictions = service.forward_pass(texts)
        for _id, pred in zip(ids, predictions):
            collection.update_one({'_id': _id}, {'$set': {'class': pred}}, upsert=False)

        pbar.update(len(ids))
    pbar.close()


