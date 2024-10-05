#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import nltk
import joblib


# In[46]:


# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = joblib.load('model.pkl')

# Setting max_review_length and tokenizer parameters
max_review_length = 500
max_features = 5000


import pickle
with open('tokenizer.pickle', 'rb') as handle:
    keras_tokenizer = pickle.load(handle)


# In[47]:


# Initialize NLTK tools
tokenizer = ToktokTokenizer()
stopword_list = set(stopwords.words('english'))


# In[48]:


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


# In[49]:


def strip_html(text):
    """Remove HTML tags from text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# In[50]:


def remove_between_square_brackets(text):
    """Remove text between square brackets."""
    return re.sub(r'\[[^]]*\]', '', text)


# In[51]:


def remove_special_characters(text, remove_digits=True):
    """Remove special characters and optionally digits."""
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


# In[52]:


def simple_stemmer(text):
    """Apply Porter stemming to each word."""
    ps = nltk.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


# In[53]:


def remove_stopwords(text, is_lower_case=False):
    """Remove stopwords from text."""
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# In[54]:


def preprocess_input_review(review):
    """Preprocess the input review like the training data."""
    # Denoise the text (remove HTML, brackets)
    review = denoise_text(review)
    
    # Remove special characters
    review = remove_special_characters(review)
    
    # Stem the words
    review = simple_stemmer(review)
    
    # Remove stopwords
    review = remove_stopwords(review)
    
    return review


# In[55]:


# Function to predict sentiment for a given review
def predict_sentiment(review):
    """Preprocess input review and predict sentiment."""
    # Preprocess the review
    review = preprocess_input_review(review)
    
    # Tokenize and pad the review
    review_sequence = keras_tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_sequence, maxlen=max_review_length)
    
    # Predict sentiment
    prediction = model.predict(review_padded)
    
    # Return positive or negative sentiment
    sentiment = "positive" if prediction >= 0.5 else "negative"
    return sentiment


# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




