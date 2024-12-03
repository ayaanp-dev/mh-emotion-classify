import pandas as pd
from sklearn import model_selection, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data(task, method):
    if task == "binary":
        return pd.read_csv(f"./new_data/binary/{method}.csv")
    elif task == "multiclass":
        return pd.read_csv(f"./new_data/multiclass/{method}.csv")
    else:
        return pd.read_csv(f"./new_data/tri/{method}.csv")

def strip_text(text):
    return text.strip()

def train_svm(task, method):
    df = load_data(task, method)
    df['content'] = df['content'].apply(strip_text)
    
    encoder = LabelEncoder()
    df['emotion'] = encoder.fit_transform(df['emotion'])

    train_x, val_x, train_y, val_y = model_selection.train_test_split(df['content'], df['emotion'], test_size=0.2, random_state=42)

    Tfidf_vect = TfidfVectorizer(max_features=5000, stop_words='english')
    Tfidf_vect.fit(df['content'])
    train_x_tfidf = Tfidf_vect.transform(train_x)
    val_x_tfidf = Tfidf_vect.transform(val_x)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_x_tfidf, train_y)
    predictions_SVM = SVM.predict(val_x_tfidf)
    print(f"Accuracy: {accuracy_score(predictions_SVM, val_y)}")

    model_save_path = f"./models/svm_{task}_{method}.pkl"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(SVM, model_save_path)
    print(f"Model saved to {model_save_path}")