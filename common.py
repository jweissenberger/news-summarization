from nltk.tokenize import sent_tokenize


def sentence_tokenizer(text):
    """
    Sentence tokenizer
    :param text:
    :return: list of sentences
    """
    text = text.replace('*', '')
    text = text.replace('-', '')
    text = text.replace('#', '')
    sentences = sent_tokenize(text)

    output = []
    for sentence in sentences:
        # remove super short sentences (usually titles or numbers or something)
        if len(sentence) < 8:
            continue
        else:
            output.append(sentence)

    return output


def article_cleaner(article):
    article = article.replace('*', '')
    article = article.replace('-', '')
    article = article.replace('#', '')

    paragraphs = article.split('\n')

    output = ""
    # This will probably remove title and small subheadings (Do I want to do this?)
    for para in paragraphs:
        if len(sentence_tokenizer(para)) == 1 or para == '' or para.isspace():
            continue
        if output == "":
            output += para



