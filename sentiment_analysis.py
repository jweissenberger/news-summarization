import flair
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def flair_sentiment_analysis(text):
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

    s = flair.data.Sentence(text)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    return total_sentiment


def textblob_sentiment_analysis(text):
    return TextBlob(text).sentiment


def nltk_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)


if __name__ == '__main__':

    file = open("cnn.txt", "r")
    cnn = file.read()
    file.close()

    file = open("fox.txt", "r")
    fox = file.read()
    file.close()

    sent = fox.split('.')[0]

    flair_sentiment_analysis(sent)
