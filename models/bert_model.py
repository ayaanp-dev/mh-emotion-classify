# train a text classification model using BERT and pytorch

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

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

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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

def train_bert(task, method):
    df = load_data(task, method)
    encoder = LabelEncoder()
    df["emotion"] = encoder.fit_transform(df["emotion"])
    df = df.rename(columns={"emotion": "label"})  # Ensure the label column is correctly named
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 512  # or any other length you prefer

    dataset = EmotionDataset(df, tokenizer, max_len)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = BertClassifier("bert-base-uncased", num_classes=len(encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
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