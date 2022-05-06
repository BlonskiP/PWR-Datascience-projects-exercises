#!/usr/bin/env python3

from bs4 import BeautifulSoup
from bs4.element import Tag

from urllib.parse import urlparse
import os

BASE_URL = "https://www.znanylekarz.pl"
QA_DOC_URL_TEMPLATE = "https://www.znanylekarz.pl/pytania-i-odpowiedzi/%s"
QA_URL_PREFIX = "https://www.znanylekarz.pl/pytania-odpowiedzi"

def parse_doctors_qa_urls(html_text):
    """
    Parse doctor q&a page urls from doctor list

    div with dp-doctor-card class
        div with media-body class
            a with znanylekarz href
    """

    dom = BeautifulSoup(html_text, "html.parser")
    cards = dom.find_all("div", class_="dp-doctor-card")
    media_bodies = [mb for card in cards for mb in card.find_all("div", class_="media-body")]
    urls = [a["href"] for mb in media_bodies for a in mb.find_all("a", href=lambda attr: attr is not None and attr.startswith(BASE_URL))]
    usernames = set()
    for url in urls:
        path = urlparse(url).path
        if len(path) == 0:
            continue
        path_components = path[1:].split("/")
        if len(path_components) != 0:
            usernames.add(path_components[0])

    urls = [QA_DOC_URL_TEMPLATE % user for user in usernames]
    return urls

def parse_qa_urls(html_text):
    """
    Parse question urls from znanylekarz.pl/pytania-i-odpowiedzi/<doctor_id>

    a tag with href starting with znanylekarz.pl/pytania-odpowiedzi
    """

    dom = BeautifulSoup(html_text, "html.parser")
    bodies = dom.find_all("p", class_="question-body")
    links = [a for body in bodies for a in body.find_all("a", href=lambda attr: attr is not None and attr.startswith(QA_URL_PREFIX))]
    urls = [link["href"] for link in links]

    return urls

def parse_qa(url, html_text):
    """
    PYTANIA I ODPOWIEDZI
    div doctor-question-body - question
    child of div doctor-answer-content - answer
    """
    QUESTION_CLASS = "doctor-question-body"
    ANSWER_CLASS = "doctor-answer-content"
    dom = BeautifulSoup(html_text, "html.parser")
    question_divs = dom.find_all("div", class_=QUESTION_CLASS)
    answer_divs = dom.find_all("div", class_=ANSWER_CLASS)

    data = []
    _, path = os.path.split(url)
    for i, div in enumerate(question_divs):
        row = {
            "id": f"q{i}_{path}",
            "url": url,
            "text": div.get_text().strip(),
            "type": "question"
        }
        data.append(row)

    for i, div in enumerate(answer_divs):
        children = [child for child in div.children if type(child) == Tag]

        for child in children:
            row = {
                "id": f"a{i}_{path}",
                "url": url,
                "text": child.get_text().strip(),
                "type": "answer"
            }
            data.append(row)

    return data
