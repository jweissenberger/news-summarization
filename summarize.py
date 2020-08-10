from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freqTable[wordValue]
                else:
                    sentenceValue[sentence] = freqTable[wordValue]

        if sentence in sentenceValue:
            sentenceValue[sentence] = sentenceValue[sentence] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.

        '''

    return sentenceValue


def _generate_summary_topn(sentences, sentenceValue, n):
    summary = ''
    important_sentences = []

    sorted_sentenceValue = sorted(sentenceValue.items(), key=lambda x: x[1])

    for i in range(1, n+1):
        important_sentences.append(sorted_sentenceValue[-i][0])

    for sentence in sentences:
        if sentence in important_sentences:
            summary += sentence + ' '

    return summary


def run_summarization(text, num_sentences):
    # Create the word frequency table
    freq_table = _create_frequency_table(text)

    # Tokenize the sentences
    sentences = sent_tokenize(text)

    # Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # take the top n important sentences
    summary = _generate_summary_topn(sentences, sentence_scores, num_sentences)

    return summary


if __name__ == '__main__':

    file = open("cnn.txt", "r")
    article = file.read()
    file.close()

    print(run_summarization(article, 10))
