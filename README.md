# Yorùbá Sentiment Analysis using Transfer Learning with AfriBERTa

This project applies transfer learning to fine-tune the AfriBERTa multilingual language model for sentiment analysis of Yorùbá text, providing a robust solution for emotion detection in African language content.

## Project Overview

This implementation uses transfer learning to adapt AfriBERTa-large (a pre-trained model with knowledge of 11 African languages, including Yorùbá) for sentiment classification. By freezing most of the base model layers and only fine-tuning specific layers, we preserve the language knowledge while adapting the model to the sentiment analysis task.

### Features

- Sentiment analysis for Yorùbá text (positive, negative, neutral)
- Transfer learning approach that preserves pre-trained language knowledge
- Interactive testing interface with real-time sentiment predictions
- Model selection from previously trained versions
- Comprehensive logging and performance metrics

## Model Architecture

- **Base Model**: AfriBERTa-large (126M parameters)
- **Architecture**: 10 layers, 6 attention heads, 768 hidden units, 3072 feed-forward size
- **Transfer Learning Approach**:
  - Frozen base layers to preserve Yorùbá language knowledge
  - Fine-tuned last 2 transformer layers for task-specific adaptation
  - New classification head for sentiment prediction (3 classes)

## Performance

After 5 epochs of training:

- **Accuracy**: 70.04%
- **F1 Score**: 69.73%
- **Precision**: 69.64%
- **Recall**: 70.04%

## Training Details

### Dataset

- Combined dataset from three sources (train, dev, test)
- Split into training (70%), validation (20%), and test (10%) sets
- Preserves class distribution through stratified sampling

### Hyperparameters

- **Learning rates**:
  - Classification head: 5e-4
  - Fine-tuned layers: 1e-5
- **Weight decay**: 0.01
- **Batch size**: 16 (with gradient accumulation of 4)
- **Number of epochs**: 10
- **Max sequence length**: 128 (reduced from 512 to speed up training)
- **Warmup steps**: 10% of total training steps
- **Gradient clipping**: 1.0

### Training Results

- **Epoch 1/10**: Training loss: 0.9543
- **Epoch 2/10**: Training loss: 0.8475
- **Epoch 3/10**: Training loss: 0.8105
- **Epoch 4/10**: Training loss: 0.7899
- **Epoch 5/10**: Training loss: 0.7640
- **Epoch 6/10**: Training loss: 0.7517
- **Epoch 7/10**: Training loss: 0.7366
- **Epoch 8/10**: Training loss: 0.7209
- **Epoch 9/10**: Training loss: 0.7159
- **Epoch 10/10**: Training loss: 0.7025

## Installation and Usage

1. **Clone the repository**:

   ```
   git clone https://github.com/Sotonye0808/ann-project.git
   cd ann-project
   ```

2. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Run the interactive test interface**:

   ```
   python user.py
   ```

   This allows you to:

   - Select from available trained models
   - Test individual Yorùbá sentences
   - Run batch evaluations on test data
   - View detailed performance metrics

4. **To retrain the model**:
   ```
   python main.py
   ```
   Training will:
   - Combine and preprocess the datasets
   - Create train/validation/test splits
   - Apply transfer learning techniques
   - Log results and save the trained model

## Data

The model was trained on a dataset of Yorùbá tweets with sentiment labels:

- **Positive**: Expressing approval, happiness, or favorable sentiment
- **Negative**: Expressing disapproval, sadness, or unfavorable sentiment
- **Neutral**: Expressing neither positive nor negative sentiment

## Project Structure

```
yoruba-sentiment-analysis/
├── main.py                # Main training script
├── train.py               # Training function with progress tracking
├── validate.py            # Validation function with metrics calculation
├── dataset.py             # Dataset handling and preprocessing
├── user.py                # Interactive testing interface with model selection
├── requirements.txt       # Project dependencies
├── afriberta_large/       # Base pre-trained model directory
├── models/                # Directory for saved fine-tuned models
├── logs/                  # Training logs and model performance records
└── datasets/              # Data directory
    └── yor_test.tsv       # Yorùbá tweets test dataset
    └── yor_train.tsv      # Yorùbá tweets training dataset
    └── yor_dev.tsv        # Yorùbá tweets development/validation dataset
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.5+
- Pandas
- NumPy
- Scikit-learn
- tqdm
- sentencepiece
- protobuf

## Technical Details

### Transfer Learning Process

1. **Base Model Preparation**:

   - Load pre-trained AfriBERTa-large model
   - Freeze most layers to preserve language knowledge
   - Add classification head for sentiment analysis

2. **Model Training**:

   - Use differential learning rates for different parts of the model
   - Apply gradient accumulation to simulate larger batch sizes
   - Implement learning rate scheduling with warmup
   - Apply gradient clipping to prevent exploding gradients

3. **Evaluation**:
   - Calculate accuracy, F1 score, precision, and recall
   - Generate confusion matrices
   - Compute class-specific metrics
   - Test on sample sentences after training

## Future Improvements

- Experiment with different sequence lengths and batch sizes
- Try different fine-tuning approaches (e.g., freezing different layers)
- Explore data augmentation techniques for minority classes
- Implement additional regularization methods

## Citations

- Muhammad, S. H., Adelani, D. I., Ruder, S., Ahmad, I. S., Abdulmumin, I., Bello, B. S., Choudhury, M., Emezue, C. C., Abdullahi, S. S., Aremu, A., Jeorge, A., & Brazdil, P. (2022). NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis. arXiv:2201.08277 [cs.CL].

- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages. In Proceedings of the 1st Workshop on Multilingual Representation Learning (pp. 116–126).
