# Sentiment-Analysis-of-Twitter

### Overview

This project demonstrates a comprehensive approach to sentiment analysis of Twitter data using the Sentiment140 dataset. By employing Natural Language Processing (NLP) techniques and machine learning, we analyze tweet sentiment and visualize insights with interactive components like sentiment scores and word clouds.

### Objective

The goal of this project is to analyze Twitter data for sentiment classification. This includes:

- Cleaning and preprocessing raw tweets.
- Visualizing data insights with word clouds and sentiment distributions.
- Training machine learning models for sentiment prediction.
- Evaluating model performance using precision, recall, F1-score, and accuracy.

### Dataset 

The Sentiment140 dataset contains 1.6 million tweets, extracted via the Twitter API and labeled for sentiment polarity.

Fields in the dataset:

- Target: Sentiment of the tweet (0 = Negative, 4 = Positive).
- IDs: Unique tweet identifier.
- Date: The timestamp of the tweet.
- Flag: Query context (e.g., NO_QUERY).
- User: Username of the person who tweeted.
- Text: Actual tweet content.

### Solution Approach

#### 1. Data Cleaning:
- Removed URLs, mentions, hashtags, and punctuation using regular expressions.
- Converted all text to lowercase for consistency.
- Applied stopword removal and lemmatization to standardize the vocabulary.

#### 2. Exploratory Data Analysis (EDA):
- Generated word clouds to identify common terms in positive and negative tweets.
- Plotted sentiment distributions to ensure balanced class representation.
- Analyzed sentiment trends over time using timestamps.

#### 3. Feature Engineering:
- Created cleaned_text for preprocessed tweet content.
- Used GloVe embeddings (100-dimensional vectors) for semantic representation of words.

#### 4. Model Building:
- Built and trained an LSTM neural network for sentiment classification.
- Experimented with baseline models like Logistic Regression and Naive Bayes using TF-IDF features for comparison.

#### 5. Evaluation:
- Evaluated models using precision, recall, F1-score, and accuracy.
- The LSTM model achieved an overall accuracy of 79%, demonstrating its effectiveness for sentiment analysis.

### Model Performance

The model achieves an overall accuracy of 79%, with balanced precision and recall across both classes.

### Results

- Sentiment Trends: Positive tweets show an upward trend over time, highlighting user satisfaction during certain periods.
- Balanced sentiment distribution across negative and positive classes.
- Common positive words include "love," "great," and "happy."
- Negative words emphasize "hate," "bad," and "sad."
- Prediction: The model reliably predicts sentiment based on tweet text with high confidence.

### Challenges

- Noisy Data: Tweets included informal language, emojis, and abbreviations.
- Class Imbalance: Addressed using balanced sampling techniques during training.
- Context Dependency: Sarcasm and negations required careful consideration to prevent misclassification.

### Source

https://www.kaggle.com/datasets/kazanova/sentiment140
