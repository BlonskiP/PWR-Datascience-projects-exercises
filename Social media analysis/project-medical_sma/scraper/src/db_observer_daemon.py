from celery_base import *
from mongo_base import *
from logger import get_logger
from tasks import classify, update_classes

from celery import chain

import argparse
import time

logger = get_logger("db_observer")

SLEEP_INTERVAL=10
BATCH_SIZE = 256

def gather_unclassified_objects(stream):
    ids = []
    texts = []
    while True:
        change = stream.try_next()
        if change is None:
            break
        identifier = change["fullDocument"]["_id"]
        text = change["fullDocument"]["tweet"]
        ids.append(identifier)
        texts.append(text)
    return ids, texts

def stream_endless_loop(stream):
    while stream.alive:
        logger.info("Checking if there are any changes...")
        
        ids, texts = gather_unclassified_objects(stream)
        
        logger.info(f"Found {len(ids)} changes")

        if len(ids) != 0:
            res = chain(classify(texts), update_classes((ids))).apply_async()

        time.sleep(SLEEP_INTERVAL)

if __name__=='__main__':
    collection = db.tweets
    query = [{"$match": {"operationType": "insert", "fullDocument.class": None}}]
    with collection.watch(query, batch_size=BATCH_SIZE) as stream:
        stream_endless_loop(stream)
