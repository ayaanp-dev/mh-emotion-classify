from nltk.stem import WordNetLemmatizer
from functools import reduce
 
lemmatizer = WordNetLemmatizer()

def lemmatize_data(df):
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    df["content"] = df["content"].apply(lambda x: reduce(lambda x, y: x + " " + lemmatizer.lemmatize(y), x, ""))
    return df