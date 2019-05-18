# -*- coding: utf-8 -*-
"""
RESOURCES:
  https://doc.scrapy.org/en/latest/intro/tutorial.html

PROCEDURE:
  $
"""
from pdb import set_trace as debug
from selenium import webdriver
import scrapy
from scrapy.selector import HtmlXPathSelector

from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


class KaggleDownloadSpider(scrapy.Spider):
    name = "kaggle_download"

    def __init__(self, download_url="https://www.kaggle.com/nih-chest-xrays/data/data"):
        self.download_url = download_url
        self.username = "xinh3ng"

    def start_requests(self):
        yield scrapy.Request(url="https://www.kaggle.com", callback=self.login)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = "quotes-%s.html" % page
        with open(filename, "wb") as f:
            f.write(response.body)
        self.log("Saved file %s" % filename)

    def get_login_cookies(self):
        driver = webdriver.Firefox()
        driver.implicitly_wait(30)
        base_url = self.start_urls[0]
        driver.get(base_url)
        driver.find_element_by_name("USER").clear()
        driver.find_element_by_name("USER").send_keys(self.username)
        driver.find_element_by_name("PASSWORD").clear()
        driver.find_element_by_name("PASSWORD").send_keys(_get_kaggle_passwd(self.username))
        driver.find_element_by_name("submit").click()
        cookies = driver.get_cookies()
        driver.close()
        return cookies

    def login(self, response):
        return [
            scrapy.FormRequest.from_response(
                response,
                formdata={"username": self.username, "password": _get_kaggle_passwd(self.username)},
                callback=self.after_login,
            )
        ]

    def after_login(self, response):
        # Check whether login is successful before going on
        if "authentication failed" in response.body:
            self.logger.error("Login failed")
        else:
            return scrapy.Request(url=self.download_url, callback=self.parse_download_url)

    def parse_download_url(self, response):
        hxs = HtmlXPathSelector(response)
        debug()
        yum = hxs.select("//img")


def _get_kaggle_passwd(username):
    debug()
