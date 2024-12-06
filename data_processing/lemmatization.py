import nltk
from nltk.stem import WordNetLemmatizer
from functools import reduce

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_data(texts):
    # Apply lemmatization to a list of texts
    lemmatized_texts = []
    for text in texts:
        lemmatized_text = reduce(lambda x, y: x + " " + lemmatizer.lemmatize(y), text.split(), "")
        lemmatized_texts.append(lemmatized_text)
    return lemmatized_texts