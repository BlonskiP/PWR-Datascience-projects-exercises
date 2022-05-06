import requests
import json
import pymongo
from celery import Celery


app = Celery()
app.conf.update(timezone = 'Europe/London')

REQUEST_CURRENT_PRICE = "https://wowtokenprices.com/current_prices.json"

MONGO_URI = 'mongodb://root:passwd@mongodb:27017'
DB_NAME = "Token_prices"
COLLECTION = "Tokens_history"
NEW_PRICES_PERIOD = 5


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(NEW_PRICES_PERIOD, request_token_price.s(REQUEST_CURRENT_PRICE),name="Request token price period")

@app.task(bind=True, name='Request',queue="default")
def request_token_price(self, request):
    print(request)
    response = requests.get(request)
    response_json = response.json()
    post_to_mongo(response_json['eu'])

def post_to_mongo(data):
    mongo_client = pymongo.MongoClient(MONGO_URI)
    db = mongo_client.Token_prices
    collection = db[COLLECTION]
    find_q =  { 'time_of_last_change_utc_timezone': data['time_of_last_change_utc_timezone'] }
    find = list(collection.find(find_q))
    print('find',find,len(find))
    if not find:
    	result = collection.insert_one(data)
    mongo_client.close()

def check_history():
    mongo_client = pymongo.MongoClient(MONGO_URI)
    db = mongo_client.Token_prices
    collection = db[COLLECTION]
    return collection.count()
