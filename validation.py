import torch
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
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
RESULTS_FILE = "validation_results.csv"
PREPROCESSED_DATA_FILE = "preprocessed_data.pt"

# Function to load the correct model based on the configuration string
def load_model(model_name, model_config, preprocess_config):
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{model_config}_{preprocess_config}.pt")
    if 'bert' == model_name.lower():
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif 'roberta' == model_name.lower():
        model = RobertaForSequenceClassification.from_pretrained("roberta-base")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError("Model name must be 'bert' or 'roberta'")

    # Load the state dictionary
    state_dict = torch.load(model_path)
    # Check if the state dictionary matches the model
    if 'bert' == model_name.lower() and 'roberta' == state_dict.keys():
        raise ValueError("Loaded state dictionary does not match the BERT model architecture.")
    if 'roberta' == model_name.lower() and 'bert' == state_dict.keys():
        raise ValueError("Loaded state dictionary does not match the RoBERTa model architecture.")
    
    model.load_state_dict(state_dict)
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

# Function to save preprocessed and tokenized data
def save_preprocessed_data(encodings, labels, filename):
    torch.save({'encodings': encodings, 'labels': labels}, filename)

# Function to load preprocessed and tokenized data
def load_preprocessed_data(filename):
    data = torch.load(filename)
    return data['encodings'], data['labels']

# Function to evaluate a model
def evaluate_model(model, tokenizer, validation_data, preprocess_config, batch_size=64):
    all_preds = []
    all_labels = []
    
    # Assuming validation_data is a DataFrame
    texts = validation_data['content'].tolist()
    labels = validation_data['emotion'].tolist()

    if os.path.exists(PREPROCESSED_DATA_FILE):
        print("Loading preprocessed and tokenized data from file...")
        encodings, labels = load_preprocessed_data(PREPROCESSED_DATA_FILE)
    else:
        print("Preprocessing and tokenizing the texts...")
        # Preprocess and tokenize the texts
        encodings = preprocess_data(texts, preprocess_config, tokenizer)
        
        # Convert labels to tensor for comparison (assuming labels are integers)
        labels = torch.tensor(labels)
        
        # Save preprocessed and tokenized data to file
        save_preprocessed_data(encodings, labels, PREPROCESSED_DATA_FILE)
    
    # Move model and data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encodings = {key: val.to(device) for key, val in encodings.items()}
    labels = labels.to(device)
    
    # Create a DataLoader for batch processing
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        print("Getting predictions from the model...")
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, batch_labels = batch
            input_ids, attention_mask, batch_labels = input_ids.to(device), attention_mask.to(device), batch_labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy}")
    return accuracy

# Function to save results to a CSV file
def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

# Example usage
if __name__ == "__main__":
    # Load validation data
    validation_data = pd.read_csv(BINARY_VALIDATION)
    
    # Define configurations to evaluate
    configurations = [
        ("bert", "binary", "lemmatization"),
        ("bert", "binary", "stemming"),
        ("bert", "binary", "universal"),
        ("roberta", "binary", "lemmatization"),
        ("roberta", "binary", "stemming"),
        ("roberta", "binary", "universal"),
        ("bert", "multiclass", "lemmatization"),
        ("bert", "multiclass", "stemming"),
        ("bert", "multiclass", "universal"),
        ("roberta", "multiclass", "lemmatization"),
        ("roberta", "multiclass", "stemming"),
        ("roberta", "multiclass", "universal"),
        ("bert", "tri", "lemmatization"),
        ("bert", "tri", "stemming"),
        ("bert", "tri", "universal"),
        ("roberta", "tri", "lemmatization"),
        ("roberta", "tri", "stemming"),
        ("roberta", "tri", "universal")
    ]
    
    # Dictionary to store results
    results = {
        "model": [],
        "config": [],
        "preprocessing": [],
        "accuracy": []
    }
    
    # Evaluate each configuration
    for model_name, model_config, preprocess_config in configurations:
        print(f"Evaluating {model_name} model with {model_config} configuration and {preprocess_config} preprocessing...")
        model, tokenizer = load_model(model_name, model_config, preprocess_config)
        accuracy = evaluate_model(model, tokenizer, validation_data, preprocess_config)
        
        # Store results
        results["model"].append(model_name)
        results["config"].append(model_config)
        results["preprocessing"].append(preprocess_config)
        results["accuracy"].append(accuracy)
    
    # Save results to a CSV file
    save_results(results, RESULTS_FILE)
