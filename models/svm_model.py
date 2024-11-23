import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

def load_data(task, method):
    if task == "binary":
        return pd.read_csv(f"./new_data/binary/{method}.csv")
    elif task == "multiclass":
        return pd.read_csv(f"./new_data/multiclass/{method}.csv")
    else:
        return pd.read_csv(f"./new_data/tri/{method}.csv")

def vectorize_data(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["content"])
    y = df["emotion"]
    return X, y

def train_svm(task, method):
    df = load_data(task, method)
    X, y = vectorize_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(SVC())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy for {task} task with {method} method: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # save model
    np.save(f"./models/svm_{task}_{method}.npy", model)

train_svm("binary", "lemmatization")