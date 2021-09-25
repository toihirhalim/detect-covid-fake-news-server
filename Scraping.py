from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests


def get_page(url):
    try:
        response = urlopen(url)
        return response.read()
    except:
        return requests.get(url).text


def get_text(url) -> str:
    page = get_page(url)
    soup = BeautifulSoup(page, features="html.parser")

    text = soup.get_text(strip=True)
    return text

