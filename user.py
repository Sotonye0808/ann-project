import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np

class YorubaSentimentAnalyzer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
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
        print(f"Predicted sentiment: {predicted_label} (confidence: {confidence:.4f})")
        
        if expected_label is not None:
            expected_label_name = self.label_map[expected_label] if isinstance(expected_label, int) else expected_label
            correct = (predicted_class.item() == expected_label) if isinstance(expected_label, int) else (predicted_label == expected_label)
            print(f"Expected sentiment: {expected_label_name}")
            print(f"Prediction {'correct' if correct else 'incorrect'} ✓" if correct else "Prediction incorrect ✗")
        
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

    def test_on_dataset(self, test_data, text_column='tweet', label_column='label'):
        """Test model on dataset and return metrics"""
        predictions = []
        true_labels = []
        
        for _, row in test_data.iterrows():
            text = row[text_column]
            expected_label = row[label_column] if isinstance(row[label_column], int) else self.label_map_inverse[row[label_column]]
            result = self.predict(text, expected_label)
            predictions.append(result['predicted_class'])
            true_labels.append(expected_label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Print results
        print("\n===== Test Dataset Results =====")
        print(f"Total samples: {len(predictions)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision, 
            'recall': recall,
            'confusion_matrix': conf_matrix
        }

def interactive_testing():
    model_path = os.path.join(os.getcwd(), "afriberta_large")
    analyzer = YorubaSentimentAnalyzer(model_path)
    
    print("==== Yorùbá Sentiment Analysis ====")
    print("Enter 'quit' to exit")
    print("Enter 'test' to run analysis on test dataset")
    
    while True:
        text = input("\nEnter Yorùbá text: ")
        if text.lower() == 'quit':
            break
        elif text.lower() == 'test':
            # Load test dataset
            data_path = os.path.join("datasets", "yor_test.tsv")
            df = pd.read_csv(data_path, sep='\t')
            label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
            df['numeric_label'] = df['label'].map(label_map)
            
            # Get test split
            from sklearn.model_selection import train_test_split
            _, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
            _, test_df = train_test_split(temp_df, test_size=0.33, random_state=42, stratify=temp_df['label'])
            
            # Run test
            analyzer.test_on_dataset(test_df, text_column='tweet', label_column='numeric_label')
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