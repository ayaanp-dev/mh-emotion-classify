# train bert and roberta models

from .bert_model import train_bert
from .roberta_model import train_roberta

def train_models():
    tasks = ["binary", "multiclass", "tri"]
    methods = ["lemmatization", "stemming", "no_preprocessing"]

    for task in tasks:
        for method in methods:
            train_bert(task, method)
            train_roberta(task, method)

train_models()