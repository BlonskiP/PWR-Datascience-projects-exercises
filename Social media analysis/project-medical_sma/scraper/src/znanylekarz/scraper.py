from znanylekarz.parser import parse_doctors_qa_urls, parse_qa_urls, parse_qa
from logger import get_logger
import requests

log = get_logger("ZNScraper")


class DoctorsNotFoundError(Exception):
    pass


class ZNScraper:
    def __init__(self, category, timebreak=100):
        self.category = category

    def _scrape_doctors(self, page):
        log.info(f"Scraping doctors for category {self.category} page {page}")
        template = "https://www.znanylekarz.pl/%s/%s"
        url = template % (self.category, page)
        response = requests.get(url)
        if response.status_code != 200:
            log.info(f"Got status code: {response.status_code}")
            return None

        html_text = response.text
        urls = parse_doctors_qa_urls(html_text)

        return urls

    def _scrape_qas_urls(self, url):
        log.info(f"Scraping doctor page for qas urls at {url}")
        response = requests.get(url)
        if response.status_code != 200:
            log.info(f"Got status code: {response.status_code}")
            return None

        html_text = response.text
        urls = parse_qa_urls(html_text)

        return urls

    def _scrape_question_and_answers(self, url):
        log.info(f"Scraping QAS for {url}")
        response = requests.get(url)
        if response.status_code != 200:
            log.info(f"Got status code: {response.status_code}")
            return []

        html_text = response.text
        qas = parse_qa(url, html_text)

        return qas

    def scrape(self, page=1):
        # page 1 seems unacceptable
        page = str(page) if page > 1 else ""

        doctor_urls = self._scrape_doctors(page)
        if doctor_urls is None:
            log.error(f"Doctors not found at page {page} for category {self.category}")
            raise DoctorsNotFoundError()

        all_qa_urls = []
        for d_url in doctor_urls:
            qa_urls = self._scrape_qas_urls(d_url)
            if qa_urls is not None:
                all_qa_urls += qa_urls
            else:
                log.info(f"Entity {d_url} probably does not have q&a section")

        if len(all_qa_urls) == 0:
            # nothing to see here
            log.info("Found nothing")
            return []

        all_qas = []
        for qa_url in all_qa_urls:
            qas = self._scrape_question_and_answers(qa_url)
            all_qas += qas
        log.info(f"Found {len(all_qas)} Q & A")

        return all_qas


__all__ = ["ZNScraper", "DoctorsNotFoundError"]
