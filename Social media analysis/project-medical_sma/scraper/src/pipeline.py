from celery_base import *
from mongo_base import *
from tasks import fetch_tweets, fetch_znanylekarz
from datetime import date, timedelta
from logger import get_logger
import sys
import os

# constants

TWITTER_SCRAPE_MODE = "tweeter"
ZNANYLEKARZ_SCRAPE_MODE = "znanylekarz"

logger = get_logger("pipeline")

# utility

def count_tweets():
    collection = db.tweets
    count = collection.count()
    logger.info(f"Tweet count: {count}")
    return count


def read_keywords(fp):
    with open(fp, "r") as f:
        keywords = [line.strip() for line in f.readlines()]

    return keywords

# pipelines

def znanylekarz_pipeline(categories_fp, page_limit=30):
    categories=read_keywords(categories_fp)

    logger.info("Fetch start!")
    responses = []
    for category in categories:
        for page in range(1, page_limit):
            logger.info(f"Fetching znanylekarz for {category} page {page}")
            res = fetch_znanylekarz(category, page).apply_async()
            responses.append(res)

    logger.info(f"Waiting for tasks {len(responses)}")
    for res in responses:
        res.get(propagate=True)


def twitter_pipeline(keywords_fp, limit=1000, limits_per_worker=200, max_lookback=1):
    TOTAL_LIMIT = limit
    KEYWORDS = read_keywords(keywords_fp) if keywords_fp is not None else None
    LIMIT_PER_WORKER = limits_per_worker
    MAX_LOOKBACK_YEARS = max_lookback

    current_date = date.today()
    oldest_possible_date = current_date - timedelta(days=365*MAX_LOOKBACK_YEARS)

    logger.info("Fetch start!")
    while count_tweets() < TOTAL_LIMIT and current_date > oldest_possible_date:
        start_date = current_date - timedelta(days=30)
        logger.info(f"Fetching tweets from: {start_date} to {current_date}")
        responses = []
        if KEYWORDS is not None:
            for word in KEYWORDS:
                res = fetch_tweets(word, start_date.isoformat(), current_date.isoformat(), LIMIT_PER_WORKER).apply_async()
                responses.append(res)
        else:
            res = fetch_tweets(None, start_date.isoformat(), current_date.isoformat(), LIMIT_PER_WORKER).apply_async()
            responses.append(res)
        current_date = start_date
        for res in responses:
            res.get(propagate=False)

# main

if __name__=='__main__':
    mode = None
    if "SCRAPE_MODE" not in os.environ:
        mode = TWITTER_SCRAPE_MODE
    else:
        mode = os.environ["SCRAPE_MODE"]


    if mode == TWITTER_SCRAPE_MODE:
        if "KEYWORDS_FILE" not in os.environ:
            logger.warning("KEYWORDS_FILE not defined. 'Scrap everything' enabled")

        keywords_fp = os.getenv("KEYWORDS_FILE")
        twitter_pipeline(keywords_fp)
    elif mode == ZNANYLEKARZ_SCRAPE_MODE:
        if "CATEGORIES_FILE" not in os.environ:
            logger.error("CATEGORIES_FILE not defined")
            exit(1)

        categories_fp = os.getenv("CATEGORIES_FILE")
        znanylekarz_pipeline(categories_fp)
