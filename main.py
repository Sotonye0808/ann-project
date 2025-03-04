# main.py
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import YorubaDataset
from train import train
from validate import validate
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load AfriBERTa model and tokenizer from local files
model_path = os.path.join(os.getcwd(), "afriberta_large")  # Path to your local model directory
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=3,  # Adjust based on sentiment classes (e.g., positive, negative, neutral)
    local_files_only=True
)
tokenizer.model_max_length = 512

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Freeze layers of the base model    
    for param in model.base_model.parameters():
        param.requires_grad = False
    # Only the classification head (and possibly the last 2 layers) will be trained
    for param in model.classifier.parameters():
        param.requires_grad = True   
    model.to(device)

    # Load Yorùbá dataset from TSV file
    data_path = os.path.join("datasets", "yor_test.tsv")
    df = pd.read_csv(data_path, sep='\t')
    
    # Map text labels to numeric values (assuming 'positive', 'negative', 'neutral')
    label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    
    # Convert labels to numeric values
    df['numeric_label'] = df['label'].map(label_map)
    
    # Split data into train/validation/test sets (70/20/10)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42, stratify=temp_df['label'])
    
    # Extract texts and labels for training
    train_texts = train_df['tweet'].tolist()
    train_labels = train_df['numeric_label'].tolist()
    
    # Extract texts and labels for validation
    val_texts = val_df['tweet'].tolist()
    val_labels = val_df['numeric_label'].tolist()
    
    # Create datasets and dataloaders
    train_dataset = YorubaDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_dataset = YorubaDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Different learning rates for different parts of the model
    # Higher learning rate for classification head, lower for last layers if unfrozen
    classifier_params = list(model.classifier.parameters())
    
    # If you decide to unfreeze last layers of base model
    last_layers_params = []
    # Unfreeze last two transformer layers
    for layer in model.base_model.encoder.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
            last_layers_params.append(param)
    
    # Different parameter groups with different learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": classifier_params, "lr": 5e-4},  # Higher learning rate for classifier
            {"params": last_layers_params, "lr": 1e-5}  # Lower learning rate for base model layers
        ],
        weight_decay=0.01
    )
    num_epochs = 5
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% of total steps for warmup
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, device, scheduler)
        print(f"Training loss: {train_loss:.4f}")

        print(f"Validation results after {num_epochs} epochs:")
        # Validate after each epoch
        accuracy, f1, precision, recall = validate(model, val_loader, device)
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    main()