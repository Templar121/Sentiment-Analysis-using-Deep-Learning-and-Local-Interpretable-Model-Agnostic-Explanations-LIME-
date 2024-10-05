import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re, string, unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import warnings
from lime.lime_text import LimeTextExplainer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')

# Load the data
imdb_data = pd.read_csv('IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)

# View sentiment distribution
print(imdb_data['sentiment'].value_counts())

# Split the dataset  
# Train dataset
train_reviews = imdb_data.review[:40000]
train_sentiments = imdb_data.sentiment[:40000]
# Test dataset
test_reviews = imdb_data.review[40000:]
test_sentiments = imdb_data.sentiment[40000:]
print(train_reviews.shape, train_sentiments.shape)
print(test_reviews.shape, test_sentiments.shape)

# Tokenization of text
tokenizer = ToktokTokenizer()
# Setting English stopwords
stopword_list = nltk.corpus.stopwords.words('english')

# Removing the HTML strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# Apply function on review column
imdb_data['review'] = imdb_data['review'].apply(denoise_text)

# Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

# Apply function on review column
imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)

# Stemming the text
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# Apply function on review column
imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)

# Set stopwords to English
stop = set(stopwords.words('english'))
print(stop)

# Removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Apply function on review column
imdb_data['review'] = imdb_data['review'].apply(remove_stopwords)

# Normalized train reviews
norm_train_reviews = imdb_data.review[:40000]
norm_train_reviews[0]

# Normalized test reviews
norm_test_reviews = imdb_data.review[40000:]
norm_test_reviews[45005]

# Count vectorizer for bag of words
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1,3))
# Transformed train reviews
cv_train_reviews = cv.fit_transform(norm_train_reviews)
# Transformed test reviews
cv_test_reviews = cv.transform(norm_test_reviews)

print('BOW_cv_train:', cv_train_reviews.shape)
print('BOW_cv_test:', cv_test_reviews.shape)
# vocab=cv.get_feature_names()-toget feature names

# Tfidf vectorizer
tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1,3))
# Transformed train reviews
tv_train_reviews = tv.fit_transform(norm_train_reviews)
# Transformed test reviews
tv_test_reviews = tv.transform(norm_test_reviews)
print('Tfidf_train:', tv_train_reviews.shape)
print('Tfidf_test:', tv_test_reviews.shape)

# Labeling the sentiment data
lb = LabelBinarizer()
# Transformed sentiment data
sentiment_data = lb.fit_transform(imdb_data['sentiment'])
print(sentiment_data.shape)

# Splitting the sentiment data
train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)

# Prepare the data for CNN
max_features = 5000
max_review_length = 500
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(norm_train_reviews)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(norm_train_reviews)
test_sequences = tokenizer.texts_to_sequences(norm_test_reviews)

# Pad sequences
train_data = pad_sequences(train_sequences, maxlen=max_review_length)
test_data = pad_sequences(test_sequences, maxlen=max_review_length)

# Convert sentiments to arrays
train_labels = np.array(train_sentiments)
test_labels = np.array(test_sentiments)

# Define and train the CNN model
model = Sequential([
    Embedding(max_features, 100, input_length=max_review_length),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {accuracy:.4f}')

# LIME for interpretability
class_names = ['negative', 'positive']
explainer = LimeTextExplainer(class_names=class_names)

def process_text(text):
    # Simple text preprocessing function
    return text.lower()

def predict_proba(arr):
    processed = []
    for i in arr:
        processed.append(process_text(i))
    list_tokenized_ex = tokenizer.texts_to_sequences(processed)
    Ex = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
    pred = model.predict(Ex)
    returnable = []
    for i in pred:
        temp = i[0]
        returnable.append(np.array([1 - temp, temp]))  # Round temp and 1-temp off to 2 places
    return np.array(returnable)

# Example instance explanation
example_index = 7574  # Use a valid index from your test data
print("Actual rating", test_sentiments[example_index])
explanation = explainer.explain_instance(norm_test_reviews.iloc[example_index], predict_proba, num_features=10)
explanation.show_in_notebook(text=True)
