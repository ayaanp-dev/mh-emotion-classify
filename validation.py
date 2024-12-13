import torch
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import os
import pandas as pd
from data_processing.lemmatization import lemmatize_data
from data_processing.stemming import stem_data
from data_processing.universal import basic_preprocessing
from tqdm import tqdm

# Define the paths to your models and validation data CSVs
MODEL_DIR = "models"
BINARY_VALIDATION = "data/imdb_binary_reviews.csv"
MULTICLASS_VALIDATION = "data/val_multiclass_data.csv"
TRICLASS_VALIDATION = "data/val_tri_data.csv"

# Function to load the correct model based on the configuration string
def load_model(model_name, model_config, preprocess_config):
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{model_config}_{preprocess_config}.pt")
    if 'bert' in model_name.lower():
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif 'roberta' in model_name.lower():
        model = RobertaForSequenceClassification.from_pretrained("roberta-base")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError("Model name must be 'bert' or 'roberta'")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, tokenizer

# Function to preprocess the validation data according to the preprocessing config
def preprocess_data(texts, preprocess_config, tokenizer):
    if preprocess_config == 'lemmatization':
        # Apply lemmatization
        texts = lemmatize_data(texts)
    elif preprocess_config == 'stemming':
        # Apply stemming
        texts = stem_data(texts)
    else:
        # Apply basic preprocessing
        texts = basic_preprocessing(texts)
        
    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    return encodings

# Function to evaluate a model
def evaluate_model(model, tokenizer, validation_data, preprocess_config):
    all_preds = []
    all_labels = []
    
    # Assuming validation_data is a DataFrame
    texts = validation_data['content'].tolist()
    labels = validation_data['emotion'].tolist()

    print("Preprocessing and tokenizing the texts...")
    # Preprocess and tokenize the texts
    encodings = preprocess_data(texts, preprocess_config, tokenizer)
    
    # Convert labels to tensor for comparison (assuming labels are integers)
    labels = torch.tensor(labels)
    
    with torch.no_grad():
        print("Getting predictions from the model...")
        # Get predictions from the model
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        # Collect predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy}")
    return accuracy

# Load the validation data CSVs
binary_validation_data = pd.read_csv(BINARY_VALIDATION)
multiclass_validation_data = pd.read_csv(MULTICLASS_VALIDATION)
triclass_validation_data = pd.read_csv(TRICLASS_VALIDATION)

# Model configurations to evaluate
configs = [
    ("bert", "binary", "lemmatization"),
    ("bert", "binary", "stemming"),
    ("bert", "binary", "preprocessing"),
    ("bert", "multiclass", "lemmatization"),
    ("bert", "multiclass", "stemming"),
    ("bert", "multiclass", "preprocessing"),
    ("bert", "tri", "lemmatization"),
    ("bert", "tri", "stemming"),
    ("bert", "tri", "preprocessing"),
    ("roberta", "binary", "lemmatization"),
    ("roberta", "binary", "stemming"),
    ("roberta", "binary", "preprocessing"),
    ("roberta", "multiclass", "lemmatization"),
    ("roberta", "multiclass", "stemming"),
    ("roberta", "multiclass", "preprocessing"),
    ("roberta", "tri", "lemmatization"),
    ("roberta", "tri", "stemming"),
    ("roberta", "tri", "preprocessing")
]

# Evaluate models and print results
for model_name, model_config, preprocess_config in configs:
    if model_config == "binary":
        validation_data = binary_validation_data
    elif model_config == "multiclass":
        validation_data = multiclass_validation_data
    else:
        validation_data = triclass_validation_data
    
    model, tokenizer = load_model(model_name, model_config, preprocess_config)
    accuracy = evaluate_model(model, tokenizer, validation_data, preprocess_config)
    print(f"Model: {model_name} | Config: {model_config} | Preprocess: {preprocess_config} | Accuracy: {accuracy:.4f}")
