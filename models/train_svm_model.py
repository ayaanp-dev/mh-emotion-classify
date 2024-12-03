# train svm model
from svm_model import train_svm

def train_models():
    tasks = ["tri", "binary", "multiclass"]
    methods = ["lemmatization", "stemming", "universal"]

    for task in tasks:
        print(f"Training {task} models")
        for method in methods:
            train_svm(task, method)

train_models()