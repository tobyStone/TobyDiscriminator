import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader
import os
import logging
import json
import spacy
from transformers import pipeline
import numpy as np
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CUDA check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Training on {device}')
logging.info(f'Still, training...')

# Character-level dataset
class DiscriminatorDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Load JSONL data
def load_data(jsonl_file):
    texts, labels = [], []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            labels.append(data['label'])
    return texts, labels

# Preprocess for character-level discriminator
def preprocess_char(texts, max_len=128):
    chars = sorted(list(set(''.join(texts))))
    char_to_idx = {char: idx+1 for idx, char in enumerate(chars)}  # Start indexing from 1
    indexed = [[char_to_idx.get(char, 0) for char in text[:max_len]] for text in texts]
    padded = [seq + [0]*(max_len-len(seq)) for seq in indexed]
    return torch.tensor(padded, dtype=torch.long), len(chars)+1

# Char-level discriminator
class CharDiscriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x).view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Syntax discriminator with SpaCy
nlp = spacy.load('en_core_web_sm')
def syntax_features(texts):
    features = []
    for doc in nlp.pipe(texts):
        pos_counts = np.zeros(len(nlp.get_pipe("tagger").labels))
        for token in doc:
            pos_counts[token.pos] += 1
        features.append(pos_counts)
    return np.array(features)

# Train syntax discriminator
def train_syntax_discriminator(X, y):
    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, 'syntax_discriminator.joblib')
    return clf

# Ensemble prediction
def ensemble_predict(text, char_model, syntax_model, hf_pipeline, char_vocab_size):
    char_input, _ = preprocess_char([text])
    char_prob = char_model(char_input.to(device)).item()

    syntax_prob = syntax_model.predict_proba(syntax_features([text]))[0][1]

    transformer_result = hf_pipeline(text)
    transformer_prob = transformer_result[0]['score'] if transformer_result[0]['label'] == 'LABEL_1' else 1 - transformer_result[0]['score']

    ensemble_score = 0.3 * char_prob + 0.3 * syntax_prob + 0.4 * transformer_prob
    return ensemble_score

# Main training function
def main(jsonl_file, model_dir, logs_dir, epochs=5, batch_size=32, lr=0.001):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    texts, labels = load_data(jsonl_file)
    labels_encoded = LabelEncoder().fit_transform(labels)

    X_char, vocab_size = preprocess_char(texts)
    X_syntax = syntax_features(texts)

    X_train_c, X_test_c, y_train, y_test = train_test_split(X_char, labels_encoded, test_size=0.2, random_state=42)
    X_train_s, X_test_s, _, _ = train_test_split(X_syntax, labels_encoded, test_size=0.2, random_state=42)

    char_dataset = DiscriminatorDataset(X_train_c, torch.tensor(y_train, dtype=torch.float32))
    char_loader = DataLoader(char_dataset, batch_size=batch_size, shuffle=True)

    char_model = CharDiscriminator(vocab_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(char_model.parameters(), lr=lr)

    # Train char-level discriminator
    char_model.train()
    for epoch in range(epochs):
        for inputs, labels in char_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = char_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        logging.info(f'Epoch {epoch+1}/{epochs} done.')

    torch.save(char_model.state_dict(), os.path.join(model_dir, 'char_discriminator.pth'))

    # Train and save syntax discriminator
    syntax_model = train_syntax_discriminator(X_train_s, y_train)

    # HuggingFace pipeline
    hf_pipeline = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

    test_text = "Sample text for ensemble prediction."
    ensemble_score = ensemble_predict(test_text, char_model, syntax_model, hf_pipeline, vocab_size)
    logging.info(f"Ensemble Score for sample: {ensemble_score}")

if __name__ == '__main__':
    main('discriminator_dataset.jsonl', 'model_dir', 'logs')
