import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from data_processing.lemmatization import lemmatize_data
from data_processing.stemming import stem_data
from data_processing.universal import basic_preprocessing
import torch
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer
from models.bert_model import BertClassifier
from models.roberta_model import RobertaClassifier

binary_accuracies = {}
multiclass_accuracies = {}
tri_accuracies = {}

binary_val = pd.read_csv("./data/imdb_binary_reviews.csv")
# positive = 1, negative = 0
binary_val["sentiment"] = binary_val["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
 
multiclass_val = pd.read_csv("./data/val_multiclass_data.csv")
tri_val = pd.read_csv("./data/val_tri_data.csv")

def test_binary():
    for method in ["lemmatization", "stemming", "universal"]:
        print(f"Testing binary with {method} preprocessing")
        bert_state_dict = torch.load(f"./models/bert_binary_{method}.pt")
        roberta_state_dict = torch.load(f"./models/roberta_binary_{method}.pt")

        bert_model = BertClassifier(pretrained_model_name='bert-base-uncased', num_classes=2)
        roberta_model = RobertaClassifier(pretrained_model_name='roberta-base', num_classes=2)

        bert_model.load_state_dict(bert_state_dict)
        roberta_model.load_state_dict(roberta_state_dict)

        bert_model.eval()
        roberta_model.eval()

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        bert_accuracies = []
        roberta_accuracies = []

        for idx, row in binary_val.iterrows():
            content = row["review"]
            label = row["sentiment"]

            bert_input = bert_tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
            roberta_input = roberta_tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)

            bert_output = bert_model(**bert_input)
            roberta_output = roberta_model(**roberta_input)

            bert_prediction = torch.argmax(bert_output).item()
            roberta_prediction = torch.argmax(roberta_output).item()

            bert_accuracies.append(bert_prediction == label)
            roberta_accuracies.append(roberta_prediction == label)

        binary_accuracies[method] = {
            "bert": sum(bert_accuracies) / len(bert_accuracies),
            "roberta": sum(roberta_accuracies) / len(roberta_accuracies)
        }

def test_multiclass():
    for method in ["lemmatization", "stemming", "universal"]:
        print(f"Testing multiclass with {method} preprocessing")
        bert_state_dict = torch.load(f"./models/bert_multiclass_{method}.pt")
        roberta_state_dict = torch.load(f"./models/roberta_multiclass_{method}.pt")

        bert_model = BertClassifier(pretrained_model_name='bert-base-uncased', num_classes=5)
        roberta_model = RobertaClassifier(pretrained_model_name='roberta-base', num_classes=5)

        bert_model.load_state_dict(bert_state_dict)
        roberta_model.load_state_dict(roberta_state_dict)

        bert_model.eval()
        roberta_model.eval()

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        bert_accuracies = []
        roberta_accuracies = []

        for idx, row in multiclass_val.iterrows():
            content = row["content"]
            label = row["emotion"]

            bert_input = bert_tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
            roberta_input = roberta_tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)

            bert_output = bert_model(**bert_input)
            roberta_output = roberta_model(**roberta_input)

            bert_prediction = torch.argmax(bert_output).item()
            roberta_prediction = torch.argmax(roberta_output).item()

            bert_accuracies.append(bert_prediction == label)
            roberta_accuracies.append(roberta_prediction == label)

        multiclass_accuracies[method] = {
            "bert": sum(bert_accuracies) / len(bert_accuracies),
            "roberta": sum(roberta_accuracies) / len(roberta_accuracies)
        }

def test_tri():
    for method in ["lemmatization", "stemming", "universal"]:
        print(f"Testing tri with {method} preprocessing")
        bert_state_dict = torch.load(f"./models/bert_tri_{method}.pt")
        roberta_state_dict = torch.load(f"./models/roberta_tri_{method}.pt")

        bert_model = BertClassifier(pretrained_model_name='bert-base-uncased', num_classes=3)
        roberta_model = RobertaClassifier(pretrained_model_name='roberta-base', num_classes=3)

        bert_model.load_state_dict(bert_state_dict)
        roberta_model.load_state_dict(roberta_state_dict)

        bert_model.eval()
        roberta_model.eval()

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        bert_accuracies = []
        roberta_accuracies = []

        for idx, row in tri_val.iterrows():
            content = row["content"]
            label = row["emotion"]

            bert_input = bert_tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
            roberta_input = roberta_tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)

            bert_output = bert_model(**bert_input)
            roberta_output = roberta_model(**roberta_input)

            bert_prediction = torch.argmax(bert_output).item()
            roberta_prediction = torch.argmax(roberta_output).item()

            bert_accuracies.append(bert_prediction == label)
            roberta_accuracies.append(roberta_prediction == label)

        tri_accuracies[method] = {
            "bert": sum(bert_accuracies) / len(bert_accuracies),
            "roberta": sum(roberta_accuracies) / len(roberta_accuracies)
        }

test_binary()
test_multiclass()
test_tri()

print("Binary accuracies")
print(binary_accuracies)
print("Multiclass accuracies")
print(multiclass_accuracies)
print("Tri accuracies")
print(tri_accuracies)