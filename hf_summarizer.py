import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
from common import sentence_tokenizer
from statistical_summarize import run_tf_idf_summarization, run_word_frequency_summarization
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM


def bart_summarize(text):
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    inputs = tokenizer([text])

    if len(inputs['input_ids'][0]) > 1024:
        del inputs
        sentences = sentence_tokenizer(text)
        return bart_summarize(run_tf_idf_summarization(text, len(sentences)-1))

    inputs = tokenizer([text], return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'])
    bart_sum = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    if type(bart_sum) is list:
        bart_sum = bart_sum[0]

    return bart_sum


def chunk_bart(text):
    """
    Chunks text into 10 sentence pieces and does bart summarize on them
    :param text:
    :return:
    """
    sentences = sentence_tokenizer(text)

    if len(sentences) > 20:
        text = run_tf_idf_summarization(text, 20)
        sentences = sentence_tokenizer(text)

    output = ''
    for chunk in divide_chunks(sentences, 10):
        part = " ".join(chunk)

        output += bart_summarize(part)

    output = output.replace('<n>', ' ')

    return output


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
    device = 'cpu'

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


def chunk_summarize_t5(text, size='large'):

    num_sentences = len(sentence_tokenizer(text))
    if num_sentences > 40:
        new_text = run_tf_idf_summarization(text, num_sentences=40)

        text = ''
        for i in new_text:
            text += i + ' '

    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained(f't5-{size}')
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    if tokenized_text.shape[1] < 500:
        return summarize_t5(text, size=size)

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


def divide_chunks(l, n):
    """Yield successive n-sized chunks from l."""

    for i in range(0, len(l), n):
        yield l[i:i + n]


def pegasus_summarization(text, model_name):

    if type(text) != list:
        text = [text]

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch(text, truncation=True, padding='longest').to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    if type(tgt_text) is list:
        temp = ''
        for i in tgt_text:
            temp += i + ''
        tgt_text = temp

    tgt_text = tgt_text.replace('<n>', ' ')

    return tgt_text
