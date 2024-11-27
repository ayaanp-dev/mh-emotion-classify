from lemmatization import lemmatize_data
from stemming import stem_data
from universal import remove_stopwords, basic_preprocessing
import pandas as pd

# lemmatization, stemming, and universal for binary, multiclass, and triclass

def load_binary_data():
    neg = pd.read_csv("./new_data/binary/negative.csv")
    pos = pd.read_csv("./new_data/binary/positive.csv")
    return pd.concat([neg, pos], ignore_index=True)

def load_multiclass_data():
    ang = pd.read_csv("./new_data/multiclass/angry.csv")
    hap = pd.read_csv("./new_data/multiclass/happiness.csv")
    lov = pd.read_csv("./new_data/multiclass/love.csv")
    neu = pd.read_csv("./new_data/multiclass/neutral.csv")
    sad = pd.read_csv("./new_data/multiclass/sadness.csv")
    return pd.concat([ang, hap, lov, neu, sad], ignore_index=True)

def load_triclass_data():
    neg = pd.read_csv("./new_data/tri/negative.csv")
    neu = pd.read_csv("./new_data/tri/neutral.csv")
    pos = pd.read_csv("./new_data/tri/positive.csv")
    return pd.concat([neg, neu, pos], ignore_index=True)

def shuffle(df):

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def preprocess_data(task, method):
    if task == "binary":
        df = load_binary_data()
    elif task == "multiclass":
        df = load_multiclass_data()
    else:
        df = load_triclass_data()

    if method == "lemmatization":
        df = shuffle(df)
        df = lemmatize_data(df)
    elif method == "stemming":
        df = shuffle(df)
        df = stem_data(df)
    else:
        df = shuffle(df)
    
    save_data(df, task, method)

def save_data(df, task, method):
    if task == "binary":
        df.to_csv(f"./new_data/binary/{method}.csv", index=False)
    elif task == "multiclass":
        df.to_csv(f"./new_data/multiclass/{method}.csv", index=False)
    else:
        df.to_csv(f"./new_data/tri/{method}.csv", index=False)

preprocess_data("binary", "lemmatization")
preprocess_data("binary", "stemming")
preprocess_data("binary", "universal")

preprocess_data("multiclass", "lemmatization")
preprocess_data("multiclass", "stemming")
preprocess_data("multiclass", "universal")

preprocess_data("triclass", "lemmatization")
preprocess_data("triclass", "stemming")
preprocess_data("triclass", "universal")