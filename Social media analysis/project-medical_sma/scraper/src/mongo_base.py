from pymongo import MongoClient
from urllib.parse import quote_plus
import os

def _mongo_url():
    try:
        url = os.environ["MONGO_URL"]
        return url
    except KeyError as error:
        print(f"Unable to read env creds: {error}")
        exit(1)

mongo = MongoClient(_mongo_url())
db = mongo.database

__all__ = ['mongo', 'db']
