from nltk.stem import PorterStemmer
from functools import reduce

def stem_data(df):
    # stemming
    ps = PorterStemmer()
    df["content"] = df["content"].apply(lambda x: reduce(lambda x, y: x + " " + ps.stem(y), x, ""))
    return df