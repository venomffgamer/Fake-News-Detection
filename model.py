import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
import requests
from urllib.parse import quote

import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of current script
data_fake = pd.read_csv(os.path.join(base_dir, "csv", "Fake.csv"), encoding="utf-8")
data_true = pd.read_csv(os.path.join(base_dir, "csv", "True.csv"), encoding="utf-8")


# Add class labels
data_fake["class"] = 0
data_true["class"] = 1

# Merge datasets
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.sample(frac=1)  # Shuffle

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)  # Fixed regex
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply preprocessing to text data
data['text'] = data['text'].apply(wordopt)

# Features and labels
x = data['text']
y = data['class']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(xv_train, y_train)
pred_lr = lr.predict(xv_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))

# API Integration for fetching real news
API_KEY = "4147c2d33c984ddfab8df3c94c4aab80"  # Replace with your actual NewsAPI key
API_HOST = "newsapi.org"

def extract_keywords(text, max_features=5):
    """Extract main keywords from text."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    vectorizer.fit([text])
    return vectorizer.get_feature_names_out()

def fetch_real_news(keywords):
    """Fetch real news from NewsAPI."""
    query = " OR ".join(keywords)
    encoded_query = quote(query)
    endpoint = f"https://{API_HOST}/v2/everything?q={encoded_query}&apiKey={API_KEY}&language=en&pageSize=5"

    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            return data.get("articles", [])
        else:
            print(f"API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Function to return label output
def output_label(n):
    return "Fake News" if n == 0 else "True News"

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test["text"])
    pred_LR = lr.predict(new_xv_test)[0]
    label = output_label(pred_LR)

    result = {"label": label, "articles": []}

    if pred_LR == 0:  # Fake News
        print("\nFetching real news articles...")
        keywords = extract_keywords(news)
        articles = fetch_real_news(keywords)
        
        if articles:  # Check if articles are found
            print("\nRelevant real news articles:")
            for article in articles:
                print(f"\nTitle: {article.get('title')}")
                print(f"Description: {article.get('description')}")
                print(f"Published At: {article.get('publishedAt')}")
                print(f"Source: {article.get('source', {}).get('name')}")
                print(f"URL: {article.get('url')}")
            
            result["articles"] = articles  # Add articles to the result dictionary
        else:
            print("No articles found.")
    else:
        print("The news is classified as True News.")
    
    return result  # Return the result containing the label and articles

# Input for manual testing
#news_input = input("Enter a news title to test: ")
#manual_testing(news_input)

