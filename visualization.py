import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string

# Load datasets dynamically based on the current file's location
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of current script
data_fake = pd.read_csv(os.path.join(base_dir, "csv", "Fake.csv"), encoding="utf-8")
data_true = pd.read_csv(os.path.join(base_dir, "csv", "True.csv"), encoding="utf-8")

# Add class labels
data_fake["class"] = 0
data_true["class"] = 1

# Merge datasets and shuffle them
data = pd.concat([data_fake, data_true], axis=0).sample(frac=1, random_state=42)

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply preprocessing
data['text'] = data['text'].apply(wordopt)

# Ensure the static/images directory exists
output_dir = os.path.join(base_dir, "static", "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=data, palette='coolwarm')
plt.title("Class Distribution")
plt.xlabel("Class (0: Fake, 1: True)")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
plt.close()

# Word Cloud for Fake News
fake_text = " ".join(data[data['class'] == 0]['text'])
wordcloud_fake = WordCloud(width=800, height=400, background_color='black').generate(fake_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Fake News")
plt.savefig(os.path.join(output_dir, 'wordcloud_fake.png'))
plt.close()

# Word Cloud for True News
true_text = " ".join(data[data['class'] == 1]['text'])
wordcloud_true = WordCloud(width=800, height=400, background_color='black').generate(true_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_true, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for True News")
plt.savefig(os.path.join(output_dir, 'wordcloud_true.png'))
plt.close()

# Text Length Distribution
data['text_length'] = data['text'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(data, x='text_length', hue='class', bins=50, kde=True, palette='coolwarm')
plt.title("Text Length Distribution by Class")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, 'text_length_distribution.png'))
plt.close()
