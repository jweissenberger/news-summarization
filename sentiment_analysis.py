import flair
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from common import sentence_tokenizer
from transformers import pipeline


def hf_sentiment_analysis(text):
    out = pipeline('sentiment-analysis')(text)

    pos_or_neg = out[0]['label']
    score = out[0]['score']

    return pos_or_neg, score


def hf_sentence_by_sentence_sentiment_analysis(text):

    sentences = sentence_tokenizer(text)

    positive_sentences = []
    negative_sentences = []

    for sent in sentences:
        positive_or_negative, score = hf_sentiment_analysis(sent)

        if positive_or_negative == 'POSITIVE':
            positive_sentences.append((score, sent))
        else:
            negative_sentences.append((score, sent))

    positive_sentences = sorted(positive_sentences, key=lambda x: x[0])
    negative_sentences = sorted(negative_sentences, key=lambda x: x[0])

    return positive_sentences, negative_sentences


def hf_topn_sentiment(text, num_sentences=3):
    pos_sentences, neg_sentences = hf_sentence_by_sentence_sentiment_analysis(text)

    top_positive = pos_sentences[-num_sentences:]
    top_negative = neg_sentences[-num_sentences:]

    return top_positive, top_negative


def flair_sentiment_analysis(text, model=None):
    """
    Measures how positive or negative the sentiment of the article is

    Seems to rate neutral sentences as positive

    :param text:
    :param model:
    :return:
    pos_or_neg: string 'POSITIVE' or 'NEGATIVE'
    score: value from 0.0 to 1 on how positive or negative
    """

    if not model:
        flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    else:
        flair_sentiment = model

    s = flair.data.Sentence(text)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    pos_or_neg = total_sentiment[0].value  # either the string 'POSITIVE' or 'NEGATIVE'
    score = total_sentiment[0].score
    return pos_or_neg, score


def flair_sentence_by_sentence_sentiment_analysis(text):

    sentences = sentence_tokenizer(text)

    positive_sentences = []
    negative_sentences = []
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

    for sentence in sentences:
        positive_or_negative, score = flair_sentiment_analysis(sentence, model=flair_sentiment)

        if positive_or_negative == 'POSITIVE':
            positive_sentences.append((score, sentence))
        else:
            negative_sentences.append((score, sentence))

    positive_sentences = sorted(positive_sentences, key=lambda x: x[0])
    negative_sentences = sorted(negative_sentences, key=lambda x: x[0])

    return positive_sentences, negative_sentences


def flair_average_sentiment(text):
    positive_sentences, negative_sentences = flair_sentence_by_sentence_sentiment_analysis(text)

    pos = 0
    neg = 0
    for i in positive_sentences:
        pos += i[0]
    pos = pos/len(positive_sentences)

    for j in negative_sentences:
        neg += j[0]
    neg = neg/len(negative_sentences)

    output = pos - neg

    return output


def flair_topn_sentiment(text, num_sentences=3):
    pos_sentences, neg_sentences = flair_sentence_by_sentence_sentiment_analysis(text)

    top_positive = pos_sentences[-num_sentences:]
    top_negative = neg_sentences[-num_sentences:]

    return top_positive, top_negative


def textblob_sentiment_analysis(text):
    """
    Measures the Polarity of text
    The polarity is a float within the range [-1.0, 1.0] where -1.0 is a very negative sentiment and 1.0 is very
    positive sentiment

    Seems to do better on neutral sentences

    :param text:
    :return:
    sentiment: float in range [-1.0, 1.0] where -1.0 is a very negative sentiment and 1.0 is very positive sentiment
    """
    return TextBlob(text).sentiment.polarity


def textblob_sentence_by_sentence_sentiment_analysis(text):

    sentences = sentence_tokenizer(text)

    output = []
    for sent in sentences:
        score = textblob_sentiment_analysis(sent)

        output.append((score, sent))

    output = sorted(output, key=lambda x: x[0])

    return output


def textblob_topn_sentiment(text, num_sentences=3):

    sentence_sentiment = textblob_sentence_by_sentence_sentiment_analysis(text)

    positive_sentences = sentence_sentiment[-num_sentences:]
    negative_sentences = sentence_sentiment[:num_sentences]

    return positive_sentences, negative_sentences


def nltk_sentiment_analysis(text):
    """

    :param text:
    :return: dict with keys 'neg'(negative), 'neu'(neutral), 'pos'(positive) and 'compound'
    Assume all are on a scale of 0-1
    compound is 0 negative to 1 positive
    """

    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)


def nltk_sentence_by_sentence_sentiment_analysis(text):

    sentences = sentence_tokenizer(text)

    output = []
    for sent in sentences:
        score = nltk_sentiment_analysis(sent)

        output.append((score, sent))

    return output


def nltk_topn_sentiment(text, num_sentences=3):

    scores_and_sentences = nltk_sentence_by_sentence_sentiment_analysis(text)

    top_positive = sorted(scores_and_sentences, key=lambda x: x[0]['pos'])[-num_sentences:]
    top_negative = sorted(scores_and_sentences, key=lambda x: x[0]['neg'])[-num_sentences:]
    top_neutral = sorted(scores_and_sentences, key=lambda x: x[0]['neu'])[-num_sentences:]

    return top_positive, top_negative, top_neutral


def run_sentiment_analysis(text, num_sentences=3):

    print("\n\n************Full Sentiment Analysis************\n\n")
    print('\n\nTechnique 1: Flair\n')
    flair_top_positive, flair_top_negative = flair_topn_sentiment(text, num_sentences=num_sentences)
    print("Top Positive Sentences:")
    for i in reversed(flair_top_positive):
        print("Score:", i[0])
        print(f"Sentence:\n{i[1]}")

    print("\n\n*********")
    print("Top Negative Sentences:")

    for i in reversed(flair_top_negative):
        print("Score:", i[0])
        print(f"Sentence:\n{i[1]}")

    print("\n\n***********************************************\n\n")
    print('Technique 2: Textblob\n')
    textb_positive_sentences, textb_negative_sentences = textblob_topn_sentiment(text, num_sentences=num_sentences)

    for i in reversed((textb_positive_sentences)):
        print("Score:", i[0])
        print(f"Sentence:\n{i[1]}")

    print("\n\n*********")
    print("Top Negative Sentences:")

    for i in textb_negative_sentences:
        print("Score:", i[0])
        print(f"Sentence:\n{i[1]}")

    print("\n\n***********************************************\n\n")
    print('Technique 3: NLTK\n')
    print("This technique returns 3 scores, Negative, Positive and Neutral\n")
    nltk_top_positive, nltk_top_negative, nltk_top_neutral = nltk_topn_sentiment(text, num_sentences=num_sentences)
    print("Top Positive Sentences:")
    for i in reversed(nltk_top_positive):
        print(f"Positive Score: {i[0]['pos']}, Negative Score: {i[0]['neg']}, Neutral Score: {i[0]['neu']}")
        print(f"Sentence:\n{i[1]}\n")

    print("\n\n*********")
    print("Top Negative Sentences:")
    for i in reversed(nltk_top_negative):
        print(f"Positive Score: {i[0]['pos']}, Negative Score: {i[0]['neg']}, Neutral Score: {i[0]['neu']}")
        print(f"Sentence:\n{i[1]}\n")

    print("\n\n*********")
    print("Top Neutral Sentences:")
    for i in reversed(nltk_top_neutral):
        print(f"Positive Score: {i[0]['pos']}, Negative Score: {i[0]['neg']}, Neutral Score: {i[0]['neu']}")
        print(f"Sentence:\n{i[1]}\n")


if __name__ == '__main__':


    file = open("fox.txt", "r")
    fox = file.read()
    file.close()

    print(hf_topn_sentiment(fox, num_sentences=3))


