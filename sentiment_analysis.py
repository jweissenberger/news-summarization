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


def hf_topn_sentiment(text, num_sentences=2):
    # check to make sure there's enough sentences
    sentences = len(sentence_tokenizer(text))
    if sentences < num_sentences*2:
        if sentences < 2:
            return ["Not enough Sentences"], ["Not enough Sentences"]
        num_sentences = 1

    pos_sentences, neg_sentences = hf_sentence_by_sentence_sentiment_analysis(text)

    top_positive = pos_sentences[-num_sentences:]
    top_negative = neg_sentences[-num_sentences:]

    return top_positive, top_negative

