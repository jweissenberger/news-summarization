from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import nltk

if __name__ == '__main__':
    #nltk
    nltk.download('vader_lexicon')
    nltk.download('subjectivity')
    nltk.download('punkt')
    nltk.download('stopwords')

    # pegasus download
    models = ['google/pegasus-cnn_dailymail']
    for model in models:
        tokenizer = PegasusTokenizer.from_pretrained(model)
        pegasus = PegasusForConditionalGeneration.from_pretrained(model)
