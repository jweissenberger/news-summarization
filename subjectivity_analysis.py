from textblob import TextBlob
from nltk.sentiment.util import demo_sent_subjectivity

from common import sentence_tokenizer


def textblob_subjectivity_analysis(text):
    """
    The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.

    Seems to do better on neutral sentences

    :param text:
    :return:
    subjectivity: float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
    """

    return TextBlob(text).sentiment.subjectivity


def textblob_sentence_by_sentence_subjectivity(text):
    sentences = sentence_tokenizer(text)

    output = []
    for sent in sentences:
        subjectivity = textblob_subjectivity_analysis(sent)

        output.append((subjectivity, sent))

    output = sorted(output, key=lambda x: x[0])

    return output


def textblob_topn_subjectivity(text, num_sentences=3):

    sentence_subjectivity = textblob_sentence_by_sentence_subjectivity(text)

    most_subjective = sentence_subjectivity[-num_sentences:]
    least_subjective = sentence_subjectivity[:num_sentences]

    return most_subjective, least_subjective


def nltk_subjectivity_analysis(text):
    # TODO
    raise NotImplementedError
