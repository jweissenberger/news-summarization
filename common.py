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
        # remove super short sentences (usually titles or numbers or something
        if len(sentence) < 8:
            continue
        else:
            output.append(sentence)

    return output