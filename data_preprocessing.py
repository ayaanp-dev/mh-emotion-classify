import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
import nltk

nltk.download("stopwords")
nltk.download('punkt_tab')

ps = PorterStemmer()

punc = r'[^\w\s]'

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = text.split()
    cleaned = [word for word in word_tokens if word not in stop_words]
    return cleaned

neg = pd.read_csv("all_negative.csv")
pos = pd.read_csv("all_positive.csv")

data = pd.concat([neg, pos])
content = data["content"].to_list()
labels = data["emotion"].to_list()
cleaned_content = []

for text in content:
    cleaned = re.sub(r"\s*[@]+\w+\s*", "", text)
    cleaned = re.sub(r"<.*?>", "", cleaned)
    cleaned = re.compile(r"https?://\S+|www\.\S+").sub(r"", cleaned)
    cleaned = remove_stopwords(cleaned)
    cleaned = text.lower()
    cleaned = re.sub(punc, "", cleaned)
    cleaned = word_tokenize(cleaned)
    cleaned = reduce(lambda x, y: x + " " + ps.stem(y), cleaned, "")
    cleaned_content.append(cleaned)

new = pd.DataFrame({"content": cleaned_content, "labels": labels})
data.to_csv("cleaned_data.csv")