# train bert and roberta models

from bert_model import train_bert
from roberta_model import train_roberta

def train_models():
    tasks = ["binary", "multiclass", "tri"]
    methods = ["lemmatization", "stemming", "universal"]

    for task in tasks:
        print(f"Training {task} models")
        for method in methods:
            print(f"Training {task} models with {method} preprocessing")
            train_bert(task, method)
            print("Finished training BERT model")
            train_roberta(task, method)

train_models()