from celery_base import *
from mongo_base import *
from logger import get_logger
from pymongo.errors import BulkWriteError

logger = get_logger("db")

@app.task(bind=True, name='save_tweets')
def save_tweets(self, tweets):
    for tweet in tweets:
        # add mongo object id
        tweet['_id'] = tweet['id']

    collection = db.tweets
    for tweet in tweets:
        if collection.find_one({'_id': tweet['_id']}) is None:
            collection.insert_one(tweet)

@app.task(bind=True, name='update_classes')
def update_classes(self, classes, ids):
    logger.info(f"Lengths: ids = {len(ids)} classes = {len(classes)}")
    logger.info(f"Updating classes for {len(ids)} tweets")
    collection = db.tweets
    for _id, _class in zip(ids, classes):
        collection.update_one({ '_id': _id }, { '$set': {'class': _class } }, upsert=False)

@app.task(bind=True, name='save_znanylekarz')
def save_znanylekarz(self, qas):
    for qa in qas:
        # add mongo object id
        qa['_id'] = qa['id']

    logger.info("Saving znanylekarz")
    collection = db.znanylekarz
    for qa in qas:
        if collection.find_one({'_id': qa['_id']}) is None:
            collection.insert_one(qa)
