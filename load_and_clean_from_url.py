from selenium import webdriver

import bs4 as bs
import urllib.request
import re


def read_and_clean_url_selenium(url):

    # get the text
    driver = webdriver.Chrome(executable_path='./chromedriver')
    driver.get(url)
    element = driver.find_element_by_tag_name('body')
    text = element.text
    driver.close()

    # clean the text
    lines = text.split('\n')

    clean_text = ""
    line_check = set()
    for line in lines:

        if len(line.split(' ')) < 6:
            continue

        # remove a line if it shows up more than once (this happens a lot for image captions and such)
        if line in line_check:
            continue
        line_check.add(line)

        clean_text += line + '\n'

    return clean_text

def _read_urllib(url):
    import requests
    page = requests.get(url)
    soup = bs.BeautifulSoup(page.content, 'html.parser')

    article = ""
    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text
        print(p.text)


if __name__ == '__main__':

    a = 0
    # TODO return the top n least important lines and prompt the user should I include these in the article
    # or something like that

    # nyt = "https://www.nytimes.com/2020/02/05/us/politics/state-of-union-speech-address.html"
    #
    # clean_text = read_and_clean_url(nyt)
    #
    # print("NYT:\n\n", run_summarization(clean_text))

    # wsj = "https://www.wsj.com/articles/top-takeaways-from-trumps-state-of-the-union-11580877144"
    #
    # clean_text = read_and_clean_url(wsj)
    # print(clean_text)
    #
    # print("\n\nWSJ:\n\n", run_summarization(clean_text))
    # cnn = 'https://www.cnn.com/2020/02/04/politics/state-of-the-union-highlights-takeaways/index.html'
    #_read_urllib(cnn)

