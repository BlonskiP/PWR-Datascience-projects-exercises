from celery import Celery, signals

app = Celery()

app.conf.task_routes = {
    "fetch_tweets": { "queue": "scraper" },
    "fetch_znanylekarz": { "queue": "scraper" },
    "save_tweets": { "queue": "db" },
    "save_znanylekarz": { "queue": "db" },
    "update_classes": { "queue": "db" },
    "classify": { "queue": "ml" }
}
