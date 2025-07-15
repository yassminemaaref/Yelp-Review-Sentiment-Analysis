from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Sample test
tokens = tokenizer.encode('I hated it , worst ', return_tensors='pt')
result = model(tokens)
print(result.logits)
print('Sentiment score:', int(torch.argmax(result.logits)) + 1)

# Scrape Yelp reviews
url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')

regex = re.compile('.*comment.*')  # May not work, dynamic class
results = soup.find_all('p', {'class': regex})

# Check if results were found
if not results:
    print("No reviews found – the class name might be incorrect or content is JS-rendered.")
else:
    reviews = [result.text for result in results]
    df = pd.DataFrame(reviews, columns=['review'])
    print(df.head())

    # Define sentiment function
    def sentiment_score(review):
        tokens = tokenizer.encode(review, return_tensors='pt', truncation=True)
        result = model(tokens)
        return int(torch.argmax(result.logits)) + 1

    # Apply sentiment analysis
    df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
    print(df[['review', 'sentiment']])
