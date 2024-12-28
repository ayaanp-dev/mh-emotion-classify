import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from models.bert_model import BertClassifier, EmotionDataset
from models.roberta_model import RobertaClassifier
from tqdm import tqdm

report_save = []

def load_validation_data(task, method):
    if task == "binary":
        return pd.read_csv("data/imdb_binary_reviews.csv")
    elif task == "multiclass":
        return pd.read_csv("data/val_multiclass_data.csv")
    else:
        return pd.read_csv(f"data/val_tri_data.csv")

def evaluate_model(model, dataloader, device):
    """Evaluates the model on the given DataLoader."""
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast():  # Use mixed precision
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    report = classification_report(val_labels, val_preds)
    return accuracy, report

def validate_model(task, method, model_type):
    df = load_validation_data(task, method)
    tokenizer_class = BertTokenizer if model_type == "bert" else RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased' if model_type == "bert" else 'roberta-base')

    df = df.rename(columns={"emotion": "label"})
    max_len = 512
    dataset = EmotionDataset(df, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)  # Set num_workers for faster data loading

    # Load model
    model_class = BertClassifier if model_type == "bert" else RobertaClassifier
    model = model_class(
        "bert-base-uncased" if model_type == "bert" else "roberta-base",
        num_classes=df["label"].nunique()
    )
    model_save_path = f"./models/{model_type}_{task}_{method}.pt"
    model.load_state_dict(torch.load(model_save_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate
    accuracy, report = evaluate_model(model, dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")

    report_save.append({
        "task": task,
        "method": method,
        "model_type": model_type,
        "accuracy": accuracy,
        "report": report
    })

def validate_all_models():
    tasks = ["binary", "multiclass", "tri"]
    methods = ["lemmatization", "stemming", "universal"]
    model_types = ["bert", "roberta"]

    for task in tasks:
        for method in methods:
            for model_type in model_types:
                validate_model(task, method, model_type)

    # Save validation results
    df = pd.DataFrame(report_save)
    df.to_csv("validation_results.csv", index=False)

if __name__ == "__main__":
    validate_all_models()