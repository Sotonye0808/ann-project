# dataset.py
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Union

class YorubaDataset(torch.utils.data.Dataset):
    """
    Dataset class for Yorùbá sentiment analysis.
    Handles tokenization and conversion of text data to tensors.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the dataset with texts, labels, and tokenizer.
        
        Args:
            texts: List of text strings to process
            labels: List of corresponding labels (integers)
            tokenizer: Pretrained tokenizer to use
            max_length: Maximum sequence length for tokenization
        """
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """
        Get the total number of items in the dataset.
        
        Returns:
            Length of the dataset
        """
        return len(self.labels)


def load_and_split_data(train_path: str, dev_path: str, test_path: str, 
                       label_map: Dict[str, int], random_state: int = 42):
    """
    Load datasets from files, combine them, and split into train/val/test sets.
    
    Args:
        train_path: Path to training data file
        dev_path: Path to development data file
        test_path: Path to test data file
        label_map: Mapping from text labels to numeric values
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df: DataFrames containing the split data
    """
    # Load datasets
    train_df = pd.read_csv(train_path, sep='\t')
    dev_df = pd.read_csv(dev_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    
    # Combine datasets
    combined_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Map labels to numeric values
    combined_df['numeric_label'] = combined_df['label'].map(label_map)
    
    # Split into train (70%), validation (20%), and test (10%) sets
    train_size = 0.7
    test_size = 0.1 / (1 - train_size)  # Adjusted to get 10% of total
    
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(combined_df, test_size=1-train_size, 
                                        random_state=random_state, stratify=combined_df['label'])
    
    # Second split: temp into validation (20% of total) and test (10% of total)
    val_df, test_df = train_test_split(temp_df, test_size=test_size, 
                                      random_state=random_state, stratify=temp_df['label'])
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df