from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from common import sentence_tokenizer


def pegasus_summarization(text, model_name):

    if type(text) != list:
        text = [text]

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch(text, truncation=True, padding='longest').to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return tgt_text


def pegasus_paragraph_by_paragraph_summary(text, model_name):

    paragraphs = text.split('\n')

    summary = ""
    for para in paragraphs:
        if len(sentence_tokenizer(para)) == 1 or para == '' or para.isspace():
            continue
        sum = pegasus_summarization(text=para, model_name=model_name)

        for i in sum:
            summary += " " + i

    return summary


def check_plagiarism(new_text, origional_text):

    # TODO need to find a better way to do this

    if type(new_text) == list:
        new = ''
        for i in new_text:
            new += i
        new_text = new

    new_text = new_text.split(' ')

    num_phrases_found = 0

    chunk_to_test = ""
    for i in range(len(new_text)):
        if i % 5 == 0 and i != 0:
            chunk_to_test += " " + new_text[i]
            if chunk_to_test in origional_text:
                num_phrases_found += 1

            chunk_to_test = ""

        else:
            chunk_to_test += " " + new_text[i]

    return num_phrases_found


if __name__ == "__main__":

    file = open("fox.txt", "r")
    article = file.read()
    file.close()

    models = ['google/pegasus-xsum', 'google/pegasus-newsroom', 'google/pegasus-cnn_dailymail',
              'google/pegasus-multi_news', 'google/pegasus-gigaword']

    import time

    from statistical_summarize import run_tf_idf_summarization, run_word_frequency_summarization

    summary_tf = run_tf_idf_summarization(article, 10)
    summary_wf = run_word_frequency_summarization(article, 10)

    for model in models:
        summary = pegasus_paragraph_by_paragraph_summary(article, model)
        print('\n\nModel:', model)
        print('Extracted phrases:', check_plagiarism())
        print(pegasus_paragraph_by_paragraph_summary(article, model))
        # a = time.time()
        # summary = pegasus_summarization(article, model_name=model)
        # b = time.time()
        #
        # print(f'\n\n\nModel: {model}\nExtracted phrases found: {check_plagiarism(summary,article)}\nTime {b - a}s\nSummary: {summary}')
        #
        # a = time.time()
        # summary = pegasus_summarization(summary_tf, model_name=model)
        # b = time.time()
        #
        # print(f'\n\n\nModel: {model}\nExtracted phrases found: {check_plagiarism(summary, article)}\nTime {b-a}s\nTF IDF Summary: {summary}')
        #
        # a = time.time()
        # summary = pegasus_paragraph_by_paragraph_summary(summary_wf, model_name=model)
        # b = time.time()
        #
        # print(f'\nModel: {model}\nExtracted phrases found: {check_plagiarism(summary, article)}\nTime {b-a}s\nWord Frequency Summary: {summary}\n\n')





