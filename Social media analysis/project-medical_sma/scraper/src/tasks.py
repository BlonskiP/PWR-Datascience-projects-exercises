from celery import signature

def fetch_znanylekarz(category, page):
    return signature("fetch_znanylekarz", args=(category, page))

def fetch_tweets(word, start_date, end_date, limit):
    return signature("fetch_tweets", args=(word, start_date, end_date, limit))

def save_tweets(tweets):
    return signature("save_tweets", args=[list(tweets)])

def update_classes(ids):
    return signature("update_classes", args=[list(ids)])

def save_znanylekarz():
    return signature("save_znanylekarz")

def classify(texts):
    return signature("classify", args=[list(texts)])
