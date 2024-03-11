import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import os
from torch.utils.data import DataLoader, TensorDataset

# Load dataset and preprocess
SENTIMENT_NAME_DIC = {'Epilepsy': 0, 'FDS': 1, 'Syncope': 2}
#data_folder = 'D:/qby/Epilepsy_prediction/text_files'
data_folder = 'D:/qby/Epilepsy_prediction/inference_results'
dataset = pd.read_csv("new_label.csv")
types = dataset.iloc[:, 1].tolist()
labels = [SENTIMENT_NAME_DIC[type] for type in types]

texts = []
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)

X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(SENTIMENT_NAME_DIC))

# Tokenize and convert to PyTorch tensors
X_train_encoded = tokenizer(X_train, padding=True, truncation=True, return_tensors='pt', max_length=200)
X_dev_encoded = tokenizer(X_dev, padding=True, truncation=True, return_tensors='pt', max_length=200)
X_test_encoded = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt', max_length=200)

y_train_tensor = torch.tensor(y_train)
y_dev_tensor = torch.tensor(y_dev)
y_test_tensor = torch.tensor(y_test)

# Create DataLoader for batching
train_data = TensorDataset(X_train_encoded['input_ids'], X_train_encoded['attention_mask'], y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Set up optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()

with torch.no_grad():
    dev_logits = model(X_dev_encoded['input_ids'].to(device), attention_mask=X_dev_encoded['attention_mask'].to(device))
    test_logits = model(X_test_encoded['input_ids'].to(device), attention_mask=X_test_encoded['attention_mask'].to(device))

dev_accuracy = (dev_logits.logits.argmax(dim=1) == y_dev_tensor.to(device)).float().mean().item()
test_accuracy = (test_logits.logits.argmax(dim=1) == y_test_tensor.to(device)).float().mean().item()

print(f"Dev accuracy: {dev_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
