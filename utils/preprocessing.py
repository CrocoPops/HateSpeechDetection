import re
import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# This downloads will be perfomed when we import preprocessing file
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

def parse_tweets_until_colon(data):
    parsed_tweets = []
    for tweet in data['text']:
        colon_index = tweet.find(':')
        if colon_index != -1:
            parsed_tweets.append(tweet[colon_index + 1:])
        else:
            parsed_tweets.append(tweet)
    return parsed_tweets

def clean_text(data):
    cleaned_tweets = []
    for tweet in data:
        tweet = tweet.lower()  # Convert text to lowercase
        tweet = re.sub(r"[^\w\s]", "", tweet)  # Remove punctuation
        tweet = re.sub(r"\d+", "", tweet)  # Remove numbers
        tweet = re.sub(r"\s+", " ", tweet).strip()  # Remove extra whitespaces
        cleaned_tweets.append(tweet)
    return cleaned_tweets

def remove_stopwords(data):
    nostopwords_tweets = []
    for tweet in data:
        tokens = word_tokenize(tweet)  # Tokenize text
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        filtered_text = " ".join(filtered_tokens)  # Join tokens back into a string
        nostopwords_tweets.append(filtered_text)
    return nostopwords_tweets

def apply_stemming(data):
    stemmer = PorterStemmer()
    stemmed_tweets = []
    for tweet in data:
        tokens = word_tokenize(tweet)  # Tokenize text
        stemmed_tokens = [stemmer.stem(token) for token in tokens]  # Apply stemming
        stemmed_text = " ".join(stemmed_tokens)  # Join tokens back into a string
        stemmed_tweets.append(stemmed_text)
    return stemmed_tweets

def apply_lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tweets = []
    for tweet in data:
        tokens = word_tokenize(tweet)  # Tokenize text
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Apply lemmatization
        lemmatized_text = " ".join(lemmatized_tokens)  # Join tokens back into a string
        lemmatized_tweets.append(lemmatized_text)
    return lemmatized_tweets

def apply_all(input_sentences):
    preprocessed_input_sentences = clean_text(input_sentences)
    preprocessed_input_sentences = remove_stopwords(preprocessed_input_sentences)
    preprocessed_input_sentences = apply_stemming(preprocessed_input_sentences)
    preprocessed_input_sentences = apply_lemmatization(preprocessed_input_sentences)
    return preprocessed_input_sentences
