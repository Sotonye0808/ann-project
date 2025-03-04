# Yorùbá Sentiment Analysis using AfriBERTa

This project applies transfer learning to fine-tune the AfriBERTa multilingual language model for sentiment analysis of Yorùbá text.

## Project Overview

This implementation uses transfer learning to adapt AfriBERTa-large (a pre-trained model with knowledge of 11 African languages, including Yorùbá) for sentiment classification.

### Features
- Sentiment analysis for Yorùbá text (positive, negative, neutral)
- Transfer learning implementation that preserves language knowledge
- Interactive testing interface for real-time sentiment predictions

## Model Architecture

- **Base Model**: AfriBERTa-large (126M parameters)
- **Architecture**: 10 layers, 6 attention heads, 768 hidden units, 3072 feed-forward size
- **Transfer Learning Approach**:
  - Frozen base layers to preserve language knowledge
  - Fine-tuned last 2 transformer layers
  - New classification head for sentiment prediction

## Performance

After 5 epochs of training:
- **Accuracy**: 0.6692
- **F1 Score**: 0.6653
- **Precision**: 0.6640
- **Recall**: 0.6692

## Training Details

### Hyperparameters
- Learning rates:
  - Classification head: 5e-4
  - Fine-tuned layers: 1e-5
- Weight decay: 0.01
- Batch size: 16
- Number of epochs: 5
- Max sequence length: 512
- Warmup steps: 10% of total training steps
- Gradient clipping: 1.0

### Training Results
- Epoch 1/5: Training loss: 0.9499
- Epoch 2/5: Training loss: 0.8194
- Epoch 3/5: Training loss: 0.7532
- Epoch 4/5: Training loss: 0.7245
- Epoch 5/5: Training loss: 0.7006

## Installation and Usage

1. **Clone the repository**:
   ```
   git clone https://github.com/Sotonye0808/ann-project.git
   cd yoruba-sentiment-analysis
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the interactive test interface**:
   ```
   python user.py
   ```

4. **To retrain the model**:
   ```
   python main.py
   ```

## Data

The model was trained on a dataset of Yorùbá tweets with sentiment labels (positive, negative, neutral).

## Project Structure
```
yoruba-sentiment-analysis/
├── main.py                # Main training script
├── train.py               # Training function
├── validate.py            # Validation function
├── dataset.py             # Dataset handling
├── user.py          # Interactive testing interface
├── requirements.txt       # Project dependencies
├── afriberta_large/       # Model directory
└── datasets/              # Data directory
    └── yor_test.tsv       # Yorùbá tweets dataset
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.5+
- Pandas
- NumPy
- Scikit-learn

## Additional
### Code Explanation
This project applies transfer learning to fine-tune AfriBERTa (a pre-trained multilingual language model) for Yorùbá sentiment analysis. The implementation:

Model Architecture: Uses AfriBERTa-large (126M parameters) as the base model

### Transfer Learning Approach:

- Freezes the base model layers to preserve language knowledge
- Unfreezes and fine-tunes only the last 2 transformer layers
- Adds a new classification head for sentiment prediction

### Dataset Processing:

- Loads Yorùbá tweets with sentiment labels from TSV file
- Splits into train (70%), validation (20%), and test (10%) sets
- Maps sentiment labels to numeric values (positive: 0, negative: 1, neutral: 2)

### Training Strategy:

- Uses differential learning rates (higher for new layers, lower for fine-tuned layers)
- Implements gradient clipping to prevent exploding gradients
- Employs a learning rate scheduler with warmup