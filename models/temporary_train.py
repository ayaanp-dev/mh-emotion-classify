# train bert and roberta models

from bert_model import train_bert
from roberta_model import train_roberta

def train_models():
    tasks = ["tri"]
    methods = ["lemmatization", "stemming", "universal"]

    for task in tasks:
        print(f"Training {task} models")
        for method in methods:
            if task == "tri" and method == "lemmatization":
                print("Skipping BERT model for tri with lemmatization preprocessing")
            else:
                print(f"Training BERT model for {task} with {method} preprocessing")
                train_bert(task, method)
                print("Finished training BERT model")

            print(f"Training RoBERTa model for {task} with {method} preprocessing")
            train_roberta(task, method)

train_models()