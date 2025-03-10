# user.py
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
from datetime import datetime

class YorubaSentimentAnalyzer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.label_map_inverse = {v: k for k, v in self.label_map.items()}
    
    def predict(self, text, expected_label=None):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        # Get results
        predicted_label = self.label_map[predicted_class.item()]
        confidence = confidence.item()
        
        # Print results
        print(f"\nText: {text}")
        print(f"Predicted sentiment: {predicted_label} (confidence: {confidence:.2%})")
        
        if expected_label is not None:
            expected_label_name = self.label_map[expected_label] if isinstance(expected_label, int) else expected_label
            correct = (predicted_class.item() == expected_label) if isinstance(expected_label, int) else (predicted_label == expected_label)
            print(f"Expected sentiment: {expected_label_name}")
            print(f"Prediction {'✓ Correct' if correct else '✗ Incorrect'}")
        
        # Return results as dictionary
        result = {
            'text': text,
            'predicted_class': predicted_class.item(),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {self.label_map[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }
        
        if expected_label is not None:
            result['expected_label'] = expected_label
            result['correct'] = correct
        
        return result

    def test_on_dataset(self, test_data, text_column='tweet', label_column='numeric_label'):
        """Test model on dataset and return metrics"""
        predictions = []
        true_labels = []
        correct_count = 0
        total_count = len(test_data)
        
        print(f"\nTesting model on {total_count} samples...")
        
        for idx, (_, row) in enumerate(test_data.iterrows()):
            text = row[text_column]
            expected_label = row[label_column]
            
            # Show progress
            if idx % 10 == 0:
                print(f"Processing sample {idx+1}/{total_count}...")
            
            result = self.predict(text, expected_label)
            predictions.append(result['predicted_class'])
            true_labels.append(expected_label)
            
            if result.get('correct', False):
                correct_count += 1
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Print results
        print("\n===== Test Dataset Results =====")
        print(f"Total samples: {total_count}")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"F1 Score: {f1:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Calculate class-wise metrics
        labels = np.unique(true_labels)
        for label in labels:
            label_name = self.label_map.get(label, str(label))
            label_precision = precision_score(true_labels, predictions, labels=[label], average=None)[0]
            label_recall = recall_score(true_labels, predictions, labels=[label], average=None)[0]
            label_f1 = f1_score(true_labels, predictions, labels=[label], average=None)[0]
            
            print(f"{label_name} class - Precision: {label_precision:.2%}, Recall: {label_recall:.2%}, F1: {label_f1:.2%}")
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision, 
            'recall': recall,
            'confusion_matrix': conf_matrix
        }

def list_available_models():
    """List available models from the models directory and their metrics from logs"""
    available_models = []
    
    # Default model (AfriBERTa)
    default_model = {
        'path': os.path.join(os.getcwd(), "afriberta_large"),
        'name': "AfriBERTa (Original pre-trained)",
        'accuracy': "Unknown",
        'timestamp': "N/A"
    }
    available_models.append(default_model)
    
    # Check if models directory exists
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        return available_models
    
    # Check if logs directory exists
    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs_dir):
        # If logs don't exist, just list model directories
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                model_info = {
                    'path': model_path,
                    'name': model_name,
                    'accuracy': "Unknown",
                    'timestamp': "Unknown"
                }
                available_models.append(model_info)
        return available_models
    
    # Get model metrics from logs
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
    model_metrics = {}
    
    for log_file in log_files:
        log_path = os.path.join(logs_dir, log_file)
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                
            model_name = log_data.get('model_name', 'Unknown')
            timestamp = log_data.get('timestamp', 'Unknown')
            accuracy = log_data.get('metrics', {}).get('accuracy', 'Unknown')
            
            model_dir = f"{model_name}_{timestamp}"
            model_path = os.path.join(models_dir, model_dir)
            
            if os.path.exists(model_path) and os.path.isdir(model_path):
                model_info = {
                    'path': model_path,
                    'name': model_name,
                    'full_name': model_dir,
                    'accuracy': accuracy,
                    'timestamp': timestamp
                }
                model_metrics[model_dir] = model_info
        except Exception as e:
            print(f"Error loading log file {log_file}: {e}")
    
    # Add model directories not found in logs
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path) and model_name not in model_metrics:
            model_info = {
                'path': model_path,
                'name': model_name,
                'accuracy': "Unknown",
                'timestamp': "Unknown"
            }
            model_metrics[model_name] = model_info
    
    # Add metrics to available_models
    for model_info in model_metrics.values():
        available_models.append(model_info)
    
    return available_models

def interactive_testing():
    print("==== Yorùbá Sentiment Analysis ====")
    
    # List available models
    available_models = list_available_models()
    print("\nAvailable models:")
    for i, model_info in enumerate(available_models):
        accuracy = model_info['accuracy']
        if isinstance(accuracy, float):
            accuracy = f"{accuracy:.2%}"
        print(f"{i+1}. {model_info['name']} (Accuracy: {accuracy}, Timestamp: {model_info['timestamp']})")
    
    # Ask user to select a model
    while True:
        try:
            model_idx = int(input("\nSelect model (enter number): ")) - 1
            if 0 <= model_idx < len(available_models):
                selected_model = available_models[model_idx]
                break
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(available_models)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Initialize analyzer with selected model
    model_path = selected_model['path']
    print(f"\nInitializing analyzer with model: {selected_model['name']}")
    analyzer = YorubaSentimentAnalyzer(model_path)
    
    print("\nEnter 'quit' to exit")
    print("Enter 'test' to run analysis on test dataset")
    print("Enter 'help' for more options")
    
    while True:
        text = input("\nEnter Yorùbá text: ")
        if text.lower() == 'quit':
            break
        elif text.lower() == 'help':
            print("\nAvailable commands:")
            print("  quit - Exit the application")
            print("  test - Run analysis on test dataset")
            print("  sample - Run analysis on 10 sample sentences from each category")
            print("  Any other text will be analyzed for sentiment")
        elif text.lower() == 'test':
            # Load test dataset
            print("\nLoading test dataset...")
            data_path = os.path.join("datasets", "yor_test.tsv")
            df = pd.read_csv(data_path, sep='\t')
            label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
            df['numeric_label'] = df['label'].map(label_map)
            
            # Get test split
            from sklearn.model_selection import train_test_split
            _, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
            _, test_df = train_test_split(temp_df, test_size=0.33, random_state=42, stratify=temp_df['label'])
            
            # Ask how many samples to test
            while True:
                try:
                    num_samples = input("\nHow many samples to test? (Enter 'all' for full test or a number): ")
                    if num_samples.lower() == 'all':
                        test_samples = test_df
                        break
                    else:
                        num_samples = int(num_samples)
                        if num_samples > 0:
                            # Stratified sampling
                            test_samples = test_df.groupby('label', group_keys=False).apply(
                                lambda x: x.sample(min(len(x), num_samples // 3))
                            )
                            break
                        else:
                            print("Please enter a positive number.")
                except ValueError:
                    print("Please enter 'all' or a valid number.")
            
            # Run test
            analyzer.test_on_dataset(test_samples, text_column='tweet', label_column='numeric_label')
        elif text.lower() == 'sample':
            # Load test dataset for sampling
            print("\nLoading samples from test dataset...")
            data_path = os.path.join("datasets", "yor_test.tsv")
            df = pd.read_csv(data_path, sep='\t')
            
            # Sample 10 sentences from each category
            samples_per_category = 10
            samples = []
            for category in ['positive', 'negative', 'neutral']:
                category_samples = df[df['label'] == category].sample(min(samples_per_category, len(df[df['label'] == category])))
                samples.append(category_samples)
            
            sample_df = pd.concat(samples)
            label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
            sample_df['numeric_label'] = sample_df['label'].map(label_map)
            
            # Run test on samples
            analyzer.test_on_dataset(sample_df, text_column='tweet', label_column='numeric_label')
        else:
            try:
                expected = input("Expected sentiment (0: positive, 1: negative, 2: neutral, or press enter to skip): ")
                expected = int(expected) if expected.strip() else None
                analyzer.predict(text, expected)
            except ValueError:
                print("Invalid input for expected sentiment. Using None.")
                analyzer.predict(text)

if __name__ == "__main__":
    interactive_testing()