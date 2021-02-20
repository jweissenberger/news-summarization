import newspaper
from newspaper import Article


def pull_articles_from_source(url, source, article_data=[]):
    paper = newspaper.build(url)
    i = 0
    failed = 0
    print(len(paper.articles))
    paper.download_articles()
    paper.parse_articles()  # remove articles that are too small (probably not articles)
    print(len(paper.articles))
    for article in paper.articles:
        i += 1
        if i > 10:
            break
        try:
            # fail if the article is empty or less than 40 words
            if article.text.isspace() or article.text == '' or len(article.text.split(' ')) < 40:
                failed += 1
                continue
            article.nlp()

            authors = article.authors
            temp = []
            for i in authors:
                if len(i.split(' ')) > 5:
                    continue
                temp.append(i)
            authors = temp

            data = {'source': source, 'title': article.title, 'authors': authors, 'text': article.text,
                    'keywords': article.keywords, 'summary': article.summary, 'url': article.url,
                    'date': article.publish_date}
            article_data.append(data)
        except:
            failed += 1

    return article_data


def source_from_url(link):

    if 'www' in link:
        source = link.split('.')[1]
    else:
        if '.com' in link:
            source = link.split('.com')[0]
        else:
            source = link.split('.')[0]
    source = source.replace('https://', '')
    source = source.replace('http://', '')
    return source


def search_for_term(terms):

    terms = terms.replace(',', '')
    terms = terms.split(' ')

    search_terms = []
    for term in terms:
        if term.isspace() or not term:
            continue

        search_terms.append(term.lower())

    urls = ['https://www.cnn.com/', 'https://www.huffpost.com/', 'https://www.msnbc.com/', 'https://www.nytimes.com/',
            'https://www.vox.com/', 'https://abcnews.go.com/', 'https://www.cbsnews.com/',
            'https://www.washingtonpost.com/', 'https://www.politico.com/', 'https://apnews.com/', 'https://www.npr.org/'
            ]
    right_urls = ['https://www.foxnews.com/', 'https://www.theamericanconservative.com/', 'https://thedispatch.com/',
                  'https://www.washingtonexaminer.com/', 'https://www.washingtonexaminer.com/', 'https://spectator.org/',
                  'https://www.theblaze.com/', 'https://www.breitbart.com/', 'https://thefederalist.com/',
                  'https://www.nationalreview.com/'
                  ]

    output_articles = []
    max_articles_per_source = 4
    for url in urls:
        matching_articles = []
        paper = newspaper.build(url)

        for article in paper.articles:
            if not article.title:
                continue
            matches = 0
            for term in search_terms:
                if term.lower() in article.title.lower():
                    matches += 1
            if matches > 0:
                matching_articles.append((article, matches))

        sorted(matching_articles, key=lambda x:x[1])

        output_articles.append(matching_articles[:max_articles_per_source])

    return output_articles


def return_single_article(link, output_type='string'):
    """

    :param link:
    :param output_type: either 'string' or 'html'
    :return:
    """

    output = {}

    source = source_from_url(link=link)
    source_names = {'foxnews': 'Fox News',
                    'brietbart': 'Brietbart',
                    'wsj': 'Wall Street Journal',
                    'cnn': 'CNN',
                    'nytimes': 'New York Times',
                    'apnews': 'The Associated Press',
                    'msnbc': 'MSNBC',
                    'washingtonpost': 'The Washington Post'}
    source = source_names.get(source, source)

    article = Article(link)

    article.download()
    article.parse()

    authors = article.authors
    temp = []
    for i in authors:
        if len(i.split(' ')) > 5:
            continue
        temp.append(i)
    authors = temp

    by_line = ''
    if len(authors) == 1:
        by_line = f'By {authors[0]}'
    else:
        by_line = 'By '
        for i in authors:
            if i == authors[-1]:
                by_line += f'and {i}'
            else:
                by_line += f'{i}, '

    if output_type == 'html':
        new_line = '<br>'
    else:
        new_line = '\n'

    results = f'{source}:{new_line}{article.title}{new_line}{by_line}{new_line}{article.text}'

    output['cleaned_article'] = results
    output['article'] = article.text
    output['title'] = article.title
    output['authors'] = by_line
    output['source'] = source
    output['url'] = link

    if output['article'].isspace() or not output['article']:
        output['article'] = "Unable to pull article from this source"

    return output
