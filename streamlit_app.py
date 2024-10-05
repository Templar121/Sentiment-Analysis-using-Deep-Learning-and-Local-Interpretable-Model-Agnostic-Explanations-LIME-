import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import pickle
import joblib

# Load your trained model and tokenizer
model = joblib.load('model.pkl')

st.set_page_config(page_title="Sentiment Analysis using LIME Explainer")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define constants
max_review_length = 500
class_names = ['negative', 'positive']
explainer = LimeTextExplainer(class_names=class_names)

def process_text(text):
    return text.lower()

def predict_proba(arr):
    processed = [process_text(i) for i in arr]
    list_tokenized_ex = tokenizer.texts_to_sequences(processed)
    Ex = pad_sequences(list_tokenized_ex, maxlen=max_review_length)
    pred = model.predict(Ex)
    return np.array([[1 - i[0], i[0]] for i in pred])

def explain_text(text):
    explanation = explainer.explain_instance(text, predict_proba, num_features=10)
    return explanation

# Inject custom CSS to make the entire Streamlit background white
def set_background_color():
    st.markdown(
        """
        <style>
        /* Make entire background white */
        .main {
            background-color: white !important;
        }
        /* Make sidebar background white */
        [data-testid="stSidebar"] {
            background-color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to apply the background color
set_background_color()

# Streamlit UI

st.markdown('<h1 style="color:white;">Sentiment Analysis with LIME Explanation</h1>', unsafe_allow_html=True)

# Input text from user
input_text = st.text_area('Enter text for sentiment analysis:')

if st.button('Analyze'):
    if input_text:
        # Display sentiment prediction
        prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences([input_text]), maxlen=max_review_length))
        sentiment = class_names[int(prediction[0][0] < 0.5)]
        st.write(f'Predicted sentiment: {sentiment}')

        # Get LIME explanation
        explanation = explain_text(input_text)

        # Display LIME explanation in HTML format
        st.write('LIME Explanation:')
        explanation_html = explanation.as_html()
        st.components.v1.html(explanation_html, height=800)

    else:
        st.write('Please enter some text.')
