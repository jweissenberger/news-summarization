import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
from common import sentence_tokenizer
from statistical_summarize import run_tf_idf_summarization, run_word_frequency_summarization


def summarize_t5(text, size='small'):
    """

    :param text:
    :param size: 'small' or 'large'
    :return:
    """

    if type(text) == list:
        new_text = ''
        for i in text:
            new_text += i + ' '

        text = new_text

    model = T5ForConditionalGeneration.from_pretrained(f't5-{size}')
    tokenizer = T5Tokenizer.from_pretrained(f't5-{size}')
    device = torch.device('cpu')

    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=100,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output


def chunk_summarize_t5(text, size='small'):


    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    device = torch.device('cpu')
    tokenizer = T5Tokenizer.from_pretrained(f't5-{size}')
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    if tokenized_text.shape[1] < 500:
        return summarize_t5(text, size=size)

    if size == 'small':
        max_size = 200
    else:
        max_size = 400

    num_chunks = math.ceil(tokenized_text.shape[1]/max_size)

    sentences = sentence_tokenizer(text)

    output = ""
    for j in chunks(sentences, num_chunks):
        output += summarize_t5(j, size=size) + ' '

    return output


def tfidf_summarize_t5(text, size='small'):

    sentences = sentence_tokenizer(text)

    if len(sentences) < 15:
        return "Article is too small for this technique"

    summary = run_tf_idf_summarization(text, num_sentences=10)

    return summarize_t5(summary, size=size)


def word_frequency_summarize_t5(text, size='small'):
    sentences = sentence_tokenizer(text)

    if len(sentences) < 15:
        return "Article is too small for this technique"

    summary = run_word_frequency_summarization(text, num_sentences=10)

    return summarize_t5(summary, size=size)


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]


if __name__ == '__main__':

    file = open("fox.txt", "r")
    text = file.read()
    file.close()

    print('\n\n\nLarge:\n', chunk_summarize_t5(text, size='large'), '\n\n\n')
    print('\n\n\nSmall:\n', chunk_summarize_t5(text, size='small'), '\n\n\n')
    print('\n\n\nLarge TFIDF:\n', tfidf_summarize_t5(text, size='large'), '\n\n\n')
    print('\n\n\nSmall TFIDF:\n', tfidf_summarize_t5(text, size='small'), '\n\n\n')
    print('\n\n\nLarge WF:\n', word_frequency_summarize_t5(text, size='large'), '\n\n\n')
    print('\n\n\nSmall WF:\n', word_frequency_summarize_t5(text, size='small'), '\n\n\n')



