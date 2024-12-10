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

### Project Workflow

##### Text Preprocessing:
- Tokenization, stopword removal, lemmatization.
- Removal of URLs, hashtags, mentions, and punctuations.

##### Visualizations:
- Sentiment distribution bar plots and pie charts.
- Word clouds for positive and negative tweets.

##### Machine Learning:
- Model: LSTM neural network.
- Word Embeddings: Pre-trained GloVe embeddings (100-dimensional).
- Evaluation Metrics: Precision, Recall, F1-score, Accuracy.

### Model Performance

The model achieves an overall accuracy of 79%, with balanced precision and recall across both classes.

### Results

- Sentiment Trends: Positive tweets show an upward trend over time, highlighting user satisfaction during certain periods.
- Balanced sentiment distribution across negative and positive classes.
- Common positive words include "love," "great," and "happy."
- Negative words emphasize "hate," "bad," and "sad."
- Prediction: The model reliably predicts sentiment based on tweet text with high confidence.

### Source

https://www.kaggle.com/datasets/kazanova/sentiment140
