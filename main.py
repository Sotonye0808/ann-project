# main.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import YorubaDataset
from train import train

# Load AfriBERTa model and tokenizer
model_name = "castorini/afriberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # Adjust based on sentiment classes (e.g., positive, negative, neutral)
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare Yorùbá dataset
    # load labeled Yorùbá text data
    train_texts = ["Your Yorùbá text examples"]
    train_labels = [0, 1, 2]  # Your corresponding labels

    # Create dataset and dataloader
    train_dataset = YorubaDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, device)

if __name__ == "__main__":
    main()