from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from common import sentence_tokenizer
import math


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


def _score_sentences_frequency(sentences, freqTable) -> dict:
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


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent] = freq_table

    return frequency_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences_tf_idf(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

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


def run_word_frequency_summarization(text, num_sentences):

    # Tokenize the sentences
    sentences = sentence_tokenizer(text)
    if len(sentences) <= num_sentences:
        return text

    # Create the word frequency table
    freq_table = _create_frequency_table(text)

    # Important Algorithm: score the sentences
    sentence_scores = _score_sentences_frequency(sentences, freq_table)

    # take the top n important sentences
    summary = _generate_summary_topn(sentences, sentence_scores, num_sentences)

    return summary


def run_tf_idf_summarization(text, num_sentences):
    """
    :param text: Plain summary_text of long article
    :return: summarized summary_text
    """

    # Sentence Tokenize
    sentences = sentence_tokenizer(text)
    if len(sentences) <= num_sentences:
        return text

    total_documents = len(sentences)

    # Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)

    # Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    # Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)

    # creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)

    # Inverse document frequency (IDF) is how unique or rare a word is.
    # Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)

    # Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)

    # Important Algorithm: score the sentences
    sentence_scores = _score_sentences_tf_idf(tf_idf_matrix)

    # Important Algorithm: Generate the summary
    summary = _generate_summary_topn(sentences, sentence_scores, num_sentences)
    return summary


def run_statistical_summarizers(text, num_sentences=10):
    print("**********Statistical Summarizations**********\n\n")

    print("TF IDF Summary:")
    print(run_tf_idf_summarization(text, num_sentences=num_sentences))

    print("\n\nWord Frequency Summary:")
    print(run_word_frequency_summarization(text, num_sentences=num_sentences), "\n\n")

