#!/usr/bin/env python3

import numpy as np
from threading import Lock
import os

from celery_base import *
from logger import get_logger

from model.embedder import Embedder
from model.classifier import Classifier
from model.dataset import purify_text

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

    def __init__(self):
        if ENV_EMBEDDING_MODEL_PATH not in os.environ:
            logger.error(f"{ENV_EMBEDDING_MODEL_PATH} not set. Failing...")
            raise ValueError("Env")
        if ENV_CLF_MODEL_PATH not in os.environ:
            logger.error(f"{ENV_CLF_MODEL_PATH} not set. Failing...")
            raise ValueError("Env")
        cuda_enabled = ENV_CUDA_ENABLED in os.environ
        self.embedder = Embedder(os.environ[ENV_EMBEDDING_MODEL_PATH], cuda=cuda_enabled, batch=4, word_embeddings=True)
        self.classifier = Classifier(os.environ[ENV_CLF_MODEL_PATH])
    
    def forward_pass(self, texts):
        logger.info(f"Purifing {len(texts)} sacret texts...")
        texts = [purify_text(t) for t in texts]
        logger.info(f"Embeddings for {len(texts)} sacret texts...")
        embeddings = self.embedder.make_embeddings(texts)
        logger.info(f"Classification for {len(texts)} sacret texts...")
        predictions = self.classifier.predict(embeddings)
        label_indices = np.argmax(predictions, axis=1)
        
        return label_indices.tolist()

@app.task(bind=True, name='classify')
def classify_sacret_text(self, texts):
    service = MLService.instance()
    result = service.forward_pass(texts)
    logger.info(f"MLService completed processing for {len(texts)} sacret texts")
    return result
