# validate.py
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from tqdm import tqdm

def validate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating model on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Calculate class-wise metrics
    labels = np.unique(all_labels)
    for label in labels:
        label_name = {0: "Positive", 1: "Negative", 2: "Neutral"}.get(label, str(label))
        label_precision = precision_score(all_labels, all_preds, labels=[label], average=None)[0]
        label_recall = recall_score(all_labels, all_preds, labels=[label], average=None)[0]
        label_f1 = f1_score(all_labels, all_preds, labels=[label], average=None)[0]
        
        print(f"{label_name} class - Precision: {label_precision:.2%}, Recall: {label_recall:.2%}, F1: {label_f1:.2%}")
    
    return accuracy, f1, precision, recall