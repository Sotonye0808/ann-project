# main.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import YorubaDataset
from train import train
from validate import validate
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import time

def load_and_combine_datasets():
    """Load and combine Yorùbá datasets from train, dev, and test files"""
    train_path = os.path.join("datasets", "yor_train.tsv")
    dev_path = os.path.join("datasets", "yor_dev.tsv")
    test_path = os.path.join("datasets", "yor_test.tsv")
    
    # Load datasets
    train_df = pd.read_csv(train_path, sep='\t')
    dev_df = pd.read_csv(dev_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    
    # Combine datasets
    combined_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Combined dataset size: {len(combined_df)} entries")
    print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    return combined_df

def sample_test_sentences(test_df, n=10):
    """Sample n sentences from each sentiment category for testing"""
    samples = []
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_df = test_df[test_df['label'] == sentiment]
        if len(sentiment_df) >= n:
            samples.extend(sentiment_df.sample(n).to_dict('records'))
        else:
            samples.extend(sentiment_df.to_dict('records'))
    
    return samples

def test_samples(model, tokenizer, samples, device, label_map):
    """Test model on sample sentences"""
    model.eval()
    results = []
    
    inverse_label_map = {v: k for k, v in label_map.items()}
    
    print("\n===== Testing Model on Sample Sentences =====")
    correct = 0
    
    for i, sample in enumerate(samples):
        text = sample['tweet']
        true_label = inverse_label_map[sample['label']]
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        predicted_label = label_map[predicted_class.item()]
        is_correct = predicted_label == sample['label']
        if is_correct:
            correct += 1
        
        # Print result
        print(f"\nSample {i+1}/{len(samples)}:")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"True sentiment: {sample['label']}")
        print(f"Predicted sentiment: {predicted_label} (confidence: {confidence.item():.2%})")
        print(f"Prediction: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        # Save result
        results.append({
            'text': text,
            'true_label': sample['label'],
            'predicted_label': predicted_label,
            'confidence': confidence.item(),
            'correct': is_correct
        })
    
    accuracy = correct / len(samples)
    print(f"\nSample Testing Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")
    
    return results, accuracy

def save_model_and_log(model, tokenizer, metrics, model_name, test_samples_results, training_info):
    """Save model and create log file with metrics"""
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and tokenizer
    print(f"\nSaving model to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Create log file with metrics
    log_data = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "sample_test_accuracy": metrics["sample_test_accuracy"]
        },        
        "training_info": training_info,  # Add training info
        "sample_test_results": test_samples_results
    }
    
    log_file = os.path.join("logs", f"{model_name}_{timestamp}.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"Model saved to {model_dir}")
    print(f"Logs saved to {log_file}")
    
    return model_dir, log_file

def main():
    print("===== Yorùbá Sentiment Analysis Model Training =====")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and combine datasets
    print("\nLoading and combining datasets...")
    combined_df = load_and_combine_datasets()
    
    # Map text labels to numeric values
    label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    
    # Convert labels to numeric values
    combined_df['numeric_label'] = combined_df['label'].map(label_map)
    
    # Split data into train/validation/test sets (70/20/10)
    print("\nSplitting data into train/validation/test sets (70/20/10)...")
    train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42, stratify=combined_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42, stratify=temp_df['label'])
    
    print(f"Training set: {len(train_df)} entries")
    print(f"Validation set: {len(val_df)} entries")
    print(f"Test set: {len(test_df)} entries")
    
    # Load AfriBERTa model and tokenizer from local files
    print("\nLoading AfriBERTa model and tokenizer...")
    model_path = os.path.join(os.getcwd(), "afriberta_large")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
        local_files_only=True
    )
    tokenizer.model_max_length = 128

    # Freeze layers of the base model    
    print("Preparing model for fine-tuning (freezing base layers)...")
    for param in model.base_model.parameters():
        param.requires_grad = False
        
    # Only the classification head and last 2 layers will be trained
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Unfreeze last two transformer layers
    print("Unfreezing last two transformer layers for fine-tuning...")
    for layer in model.base_model.encoder.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    model.to(device)

    # Extract texts and labels for training
    train_texts = train_df['tweet'].tolist()
    train_labels = train_df['numeric_label'].tolist()
    
    # Extract texts and labels for validation
    val_texts = val_df['tweet'].tolist()
    val_labels = val_df['numeric_label'].tolist()
    
    train_batch_size = 8  # Batch size
    gradient_accumulation_steps = 4  # Accumulate gradients

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = YorubaDataset(train_texts, train_labels, tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    val_dataset = YorubaDataset(val_texts, val_labels, tokenizer, max_length=128)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)

    # Different learning rates for different parts of the model
    classifier_params = list(model.classifier.parameters())
    
    # Get parameters from unfrozen layers
    last_layers_params = []
    for layer in model.base_model.encoder.layer[-2:]:
        for param in layer.parameters():
            if param.requires_grad:
                last_layers_params.append(param)
    
    # Different parameter groups with different learning rates
    print("Setting up optimizer with differential learning rates...")
    optimizer = torch.optim.AdamW(
        [
            {"params": classifier_params, "lr": 5e-4},  # Higher learning rate for classifier
            {"params": last_layers_params, "lr": 1e-5}  # Lower learning rate for base model layers
        ],
        weight_decay=0.01
    )
    num_epochs = 5
    
    # Learning rate scheduler
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),  # 5% warmup
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n===== Starting Training =====")
    print("\n--Capturing Training Info--")
    training_info = {
        "epochs": [],
        "total_training_time": 0,
        "hyperparameters": {
            "batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rates": {
                "classifier": 5e-4,
                "fine_tuned_layers": 1e-5
            },
            "weight_decay": 0.01,
            "sequence_length": 128,
            "warmup_ratio": 0.05
        }
    }

    training_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, device, scheduler, gradient_accumulation_steps=gradient_accumulation_steps)
        
        epoch_training_time = time.time() - epoch_start_time
        print(f"Training loss: {train_loss:.4f}")
        print(f"Epoch training time: {epoch_training_time:.2f} seconds ({epoch_training_time/60:.2f} minutes)")

        # Validate after each epoch
        print("\nRunning validation...")
        val_start_time = time.time()
        accuracy, f1, precision, recall = validate(model, val_loader, device)
        val_time = time.time() - val_start_time
        
        # Convert metrics to percentages for display
        print(f"Validation Results:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"F1 Score: {f1:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"Validation time: {val_time:.2f} seconds ({val_time/60:.2f} minutes)")
        
        # Store epoch information
        training_info["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "training_time": epoch_training_time,
            "validation_time": val_time
        })
    
    training_info["total_training_time"] = time.time() - training_start_time
    

    # Test on sample sentences from test set
    print("\nSampling test sentences for final evaluation...")
    test_samples_data = sample_test_sentences(test_df)
    test_samples_results, sample_test_accuracy = test_samples(model, tokenizer, test_samples_data, device, {v: k for k, v in label_map.items()})
    
    # Save model and create log
    model_name = "yoruba_sentiment_model"
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sample_test_accuracy": sample_test_accuracy
    }
    
    model_dir, log_file = save_model_and_log(model, tokenizer, metrics, model_name, test_samples_results, training_info)
    
    print("\n===== Training Complete =====")
    print(f"Final Validation Accuracy: {accuracy:.2%}")
    print(f"Sample Test Accuracy: {sample_test_accuracy:.2%}")
    print(f"Model saved to: {model_dir}")
    print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    main()