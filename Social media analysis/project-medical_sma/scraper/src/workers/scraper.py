#!/usr/bin/env python3

from celery_base import *
from celery import chain, group
from prometheus_client import Summary, Counter, Gauge
from tasks import save_tweets, save_znanylekarz, classify, update_classes
from logger import get_logger
import twint
from znanylekarz.scraper import ZNScraper


TWEET_COUNTER = Counter('scraper_tweets_count', 'Number of scraped tweets')
PROCESSING_TIME = Summary('scraper_processing_seconds', 'Time spent processing')
WORKERS_RUNNING = Gauge('scraper_inprogress_workers', 'Workers')

POLISH_LANGUAGE_CODE = 'pl'

logger = get_logger("scraper")


@app.task(bind=True, name='fetch_tweets')
@PROCESSING_TIME.time()
def fetch_tweets(self, word, start_date, end_date, limit):
    WORKERS_RUNNING.inc()

    # create config and search
    c = twint.Config()
    c.Limit = limit
    c.Lang = POLISH_LANGUAGE_CODE
    c.Hide_output = True
    c.Store_object = True # store in memory
    c.Popular_tweets = True
    c.Min_likes = 2
    if word is not None:
        c.Search = word
    if start_date is not None:
        # start date is string here in yyyy-mm-dd
        c.Since = start_date
    if end_date is not None:
        # end date is string here in yyyy-mm-dd
        c.Until = end_date
    
    twint.run.Search(c)
    tweets = twint.output.tweets_list
    tweets_dicts = [ vars(t) for t in tweets ]
    texts = [ t.tweet for t in tweets ]
    ids = [ t.id for t in tweets ]

    # update stats
    TWEET_COUNTER.inc(len(tweets))
    WORKERS_RUNNING.dec()

    # spawn task to save tweets
    clf_sig = chain(classify(texts), update_classes(ids))
    save_sig = save_tweets(tweets_dicts) 
    res = group(save_sig, clf_sig).apply_async(propagate=False)
    # free resources
    #res.forget()

@app.task(bind=True, name='fetch_znanylekarz')
def fetch_znanylekarz(self, category, page):
    WORKERS_RUNNING.inc()

    try:
        scraper = ZNScraper(category)
        qas = scraper.scrape(page)

        logger.info(f"Saving {len(qas)}")
        res = save_znanylekarz().apply_async(args=[list(qas)], propagate=False)

        # free resources
        # res.forget()
    except Exception as e:
        logger.info(e)
