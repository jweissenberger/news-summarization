# News Summarization

This repo has code to scrape and summarize news articles. This repo uses multiple ML models through huggingface and 
two statistical techniques to summarize text. It also uses newspaper3k to scrape the articles. Look at the demo notebook 
to see how it works 

### Set up
`pip install -r requirements.txt`

`python models_download.py` # this is to download the necessary huggingface models and nltk datasets

```python
from scraping import return_single_article
article = return_single_article('https://www.cnn.com/2020/10/20/politics/joe-biden-tax-plan/index.html')

from hf_summarizer import bart_summarize
summary = bart_summarize(article['article'])

print(summary)
>>> """Democratic presidential candidate Joe Biden has put forth several proposals that would change the tax code. 
He's proposing to raise taxes on the wealthy and on corporations by reversing some of the Republican-backed tax cuts 
signed into law in 2017. It's unlikely that Biden's campaign plans would come to fruition just as he's proposed them, 
even if he wins the election."""

```

There is additional sentiment analysis code and other useful functions in the repo such as plagiarism detection and 
subjectivity analysis. 


My repo `jweissenberger/newsletter` wraps this code in an easy to use webapp
