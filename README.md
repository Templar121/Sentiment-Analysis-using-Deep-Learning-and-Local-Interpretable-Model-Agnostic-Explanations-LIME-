# IMDB Sentiment Analysis with CNN and LIME

This project performs sentiment analysis on IMDB movie reviews using Convolutional Neural Networks (CNN) and provides model interpretability with LIME (Local Interpretable Model-agnostic Explanations).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Interpretability with LIME](#interpretability-with-lime)
8. [Usage](#usage)
9. [License](#license)

## Project Overview
This project aims to classify IMDB movie reviews as either "positive" or "negative" sentiment using a deep learning-based Convolutional Neural Network (CNN). The model is trained on the IMDB dataset, and LIME is used for model interpretability to understand which words contribute the most to predictions.

## Dependencies
The following libraries are required to run the code:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `nltk`
- `scikit-learn`
- `textblob`
- `tensorflow`
- `lime`
- `beautifulsoup4`
- `wordcloud`
- `spacy`

To install the necessary dependencies, you can use the following command:

```bash
pip install numpy pandas seaborn matplotlib nltk scikit-learn textblob tensorflow lime beautifulsoup4 wordcloud spacy
```
---
## Data Preprocessing

The IMDB dataset consists of movie reviews and corresponding sentiment labels. The following preprocessing pipeline is applied to the reviews before training the model:

1. **HTML Stripping:** 
   - HTML tags are removed using the BeautifulSoup library to clean up the raw review text.

2. **Text Cleaning:** 
   - Unnecessary characters, such as square brackets and special characters, are removed using regular expressions.

3. **Tokenization:** 
   - The text is split into individual words (tokens) using NLTK's `ToktokTokenizer` to process the review at the word level.

4. **Stopword Removal:** 
   - Common stopwords (e.g., "and", "the", "is", "in") are removed using NLTK's list of English stopwords. This helps in reducing noise and focusing on meaningful words.

5. **Stemming:** 
   - Words are reduced to their base form using the Porter stemmer (e.g., "running" becomes "run").

6. **Text Vectorization:** 
   - The preprocessed text data is then converted into numerical form using two methods:
     - **Bag-of-Words (BOW):** A simple technique that represents text as a collection of words and their frequencies.
     - **TF-IDF (Term Frequency-Inverse Document Frequency):** A more advanced technique that considers the importance of a word based on its frequency in the document and across the corpus.

These steps ensure the text data is cleaned and represented numerically for model training.
---

## Model Architecture

A **Convolutional Neural Network (CNN)** is used for sentiment analysis in this project. The model is designed to learn spatial patterns in text sequences and classify sentiment based on the learned features. The architecture includes the following layers:

1. **Embedding Layer:** 
   - This layer converts the words in the input text into dense vector representations. The vectors capture semantic relationships between words.

2. **Convolutional Layer:** 
   - A convolutional layer is applied to the embedded text data to learn spatial patterns and local dependencies in the text sequences. The kernel size used is 5, which means the convolutional operation looks at sequences of 5 words at a time.

3. **Global Max Pooling:** 
   - After the convolution operation, global max pooling is applied to reduce the dimensionality of the output and retain the most important features learned by the convolutional layer.

4. **Fully Connected Layers:** 
   - These layers are used to perform the final classification. They process the features learned from the previous layers to determine whether the sentiment of the review is positive or negative.

5. **Dropout Layer:** 
   - A dropout layer is included to prevent overfitting. It randomly sets a fraction of the input units to 0 during training, helping the model generalize better.

The CNN model is compiled using the following settings:
- **Optimizer:** Adam optimizer, known for its adaptive learning rate.
- **Loss Function:** Binary cross-entropy, suitable for binary classification problems like sentiment analysis.

This architecture allows the model to effectively capture patterns in the text and make accurate sentiment predictions.
---

## Training

The model is trained using the training data consisting of **40,000 reviews**. During training, the following parameters are used:
- **Validation Split:** 20% of the training data is reserved for validation to monitor the model's performance on unseen data during training.
- **Epochs:** The model is trained for **5 epochs** to allow enough iterations for the model to learn meaningful features.
- **Batch Size:** The batch size is set to **64**, which determines how many samples are processed before the model's weights are updated.

The model's training process helps it learn to predict sentiment from the reviews effectively.
---

## Evaluation

The model’s performance is evaluated on a **test set** consisting of **10,000 reviews**. The evaluation metric used is **accuracy**, which measures the proportion of correctly predicted sentiments. After the evaluation, the results (test accuracy) are printed to the console, providing insight into how well the model generalizes to unseen data.
---

## Interpretability with LIME

*LIME (Local Interpretable Model-agnostic Explanations)** is used to interpret the model's predictions for individual reviews. It provides insight into which features (words) were most influential in determining the sentiment of a given review. This is particularly useful for understanding and explaining black-box model decisions.
---

### To use LIME:

1. **LimeTextExplainer:** 
   - The `LimeTextExplainer` class from the `lime` library is used to explain text-based predictions. It can interpret the model's decisions on individual test instances.
   
2. **Prediction Probabilities:** 
   - The model’s prediction probabilities are provided to the explainer. These probabilities help LIME understand how confident the model is about its predictions.

3. **Explain Predictions:** 
   - Predictions for a specific test instance are explained, showing the most important words in the review that influenced the sentiment prediction.

LIME helps in making the model's decisions more transparent and interpretable, which is crucial for trust in real-world applications.
