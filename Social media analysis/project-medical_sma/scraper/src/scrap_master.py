#!/usr/bin/env python3
#
# Simple command line app to order some scrapping with parameters of choice.
#
# Usage:
# scrap_master 
#

from celery_base import app as celery_app
from tasks import fetch_tweets
from art import tprint
from datetime import date
import time
import math
from tqdm import tqdm

def get_words():
    words = []
    print("[*] Type in keywords to scrap. Press return after each word. Leave input empty to stop.")
    while True:
        word = input(f"Word {len(words)}: ")
        if len(word) != 0:
            words.append(word)
        else:
            break
    answer = input(f"[*] Chosen words: {words}. Are you sure, these are words you want (y/n)?: ")
    if answer.starts_with("y"):
        return words 
    else:
        return None

def get_daterange():
    start_date = None
    end_date = None
    print("[*] Pick tweet creation date range. Expected date format yyyy-MM-dd")
    while start_date is None:
        try:
            sd_input = input("Start date: ")
            start_date = date.fromisoformat(sd_input)
        except ValueError:
            print("Invalid date. Try again")
    while end_date is None:
        try:
            ed_input = input("End date: ")
            end_date = date.fromisoformat(ed_input)
        except ValueError:
            print("Invalid date. Try again")
    return start_date, end_date

def get_integer(prompt):
    number = None
    while number is None:
        try:
            number = int(input(prompt))
        except ValueError:
            print("This is not a number. Try again")
    return number

def get_task_count():
    return get_integer("[*] Choose number of workers per word: ")

def get_limit():
    return get_integer("[*] Chose scrap limit per worker: ")

def split_date_range(start_date, end_date, steps):
    start_epoch = int(time.mktime(start_date.timetuple()))
    end_epoch = int(time.mktime(end_date.timetuple()))
    step = math.ceil((end_epoch - start_epoch) / steps)
    for i_epoch in range(start_epoch, end_epoch, step):
        i_end_epoch = (i_epoch + step)
        if i_end_epoch > end_epoch:
            i_end_epoch = end_epoch
        yield date.fromtimestamp(i_epoch), date.fromtimestamp(i_end_epoch)

if __name__ == '__main__':   
    tprint("Scrap Master")
    print("[*] Let's scrap some shit!")

    words = get_words()
    if words is None:
        exit(0)
    start_date, end_date = get_daterange()
    count = get_task_count()
    limit = get_limit()

    for word in words:
        for begin, end in split_date_range(start_date, end_date, count):
            res = fetch_tweets(word, begin, end, limit)
            # dont wait!
            res.forget()