# train a text classification model using RoBERTa and pytorch

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Ensure the DataFrame has the 'label' column
        if 'label' not in self.df.columns:
            raise KeyError("DataFrame must contain a 'label' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        content = str(self.df.iloc[idx]["content"])
        inputs = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs.get('token_type_ids', torch.tensor([0] * self.max_len)).squeeze()
        label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label
        }

class RobertaClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_data(task, method):
    if task == "binary":
        return pd.read_csv(f"./new_data/binary/{method}.csv")
    elif task == "multiclass":
        return pd.read_csv(f"./new_data/multiclass/{method}.csv")
    else:
        return pd.read_csv(f"./new_data/{method}.csv")

def train_roberta(task, method):
    df = load_data(task, method)
    print("data loaded")
    encoder = LabelEncoder()
    df["emotion"] = encoder.fit_transform(df["emotion"])
    df = df.rename(columns={"emotion": "label"})  # Ensure the label column is correctly named
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = 512  # or any other length you prefer

    dataset = EmotionDataset(df, tokenizer, max_len)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increased batch size
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Increased batch size
    model = RobertaClassifier("roberta-base", num_classes=len(encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    accumulation_steps = 4  # Number of batches to accumulate gradients
    scaler = GradScaler()

    num_epochs = 1
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=False)
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            
            with autocast():
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps  # Normalize loss
            
            scaler.scale(loss).backward()
            train_loss += loss.item()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            progress_bar.set_postfix({"loss": train_loss / (i + 1)})
        
        model.eval()
        val_preds = []
        val_labels = []
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask, token_type_ids)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy_score(val_labels, val_preds)}")
        print(classification_report(val_labels, val_preds))

    model_save_path = f"./models/roberta_{task}_{method}.pt"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")