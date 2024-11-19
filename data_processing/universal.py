import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import reduce
import nltk

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = text.split()
    cleaned = [word for word in word_tokens if word not in stop_words]
    return cleaned

def basic_preprocessing(df):
    # lowecasing, removal of punctuation and special characters, removal of stop-words, removal of urls and html tags, tokenization
    punc = r'[^\w\s]'
    cleaned_content = []
    for text in df["content"]:
        cleaned = re.sub(r"\s*[@]+\w+\s*", "", text)
        cleaned = re.sub(r"<.*?>", "", cleaned)
        cleaned = re.compile(r"https?://\S+|www\.\S+").sub(r"", cleaned)
        cleaned = remove_stopwords(cleaned)
        cleaned = text.lower()
        cleaned = re.sub(punc, "", cleaned)
        cleaned = word_tokenize(cleaned)
        cleaned_content.append(cleaned)
    df["content"] = cleaned_content
    return df