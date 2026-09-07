# 🎬 SentimentScope Transformer ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-orange)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)]()
[![Transformer](https://img.shields.io/badge/Model-Custom%20Transformer-purple)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## Overview

SentimentScope is an end-to-end NLP sentiment classification pipeline built with Python and PyTorch.

The project implements a custom Transformer architecture for binary sentiment classification of IMDB movie reviews. Reviews are tokenized using the `bert-base-uncased` tokenizer and classified as either positive or negative.

The final model achieved **76.49% test accuracy**, exceeding the project requirement of 75%.

---

## Business Context

The project is framed around CineScope, an entertainment recommendation platform seeking to improve personalization by understanding user sentiment toward movies and shows.

Sentiment analysis can support recommendation systems by identifying whether users express positive or negative opinions about content.

---

## ML Pipeline

```text
IMDB Reviews
     │
     ▼
Data Exploration
     │
     ▼
Train / Validation Split
     │
     ▼
BERT Subword Tokenizer
     │
     ▼
Custom PyTorch Dataset
     │
     ▼
DataLoader
     │
     ▼
Custom Transformer
     │
     ▼
Mean Pooling
     │
     ▼
Classification Head
     │
     ▼
Positive / Negative
```

---

## Transformer Architecture

```text
Token IDs
   │
   ▼
Token Embeddings
   +
Positional Embeddings
   │
   ▼
┌─────────────────────────┐
│ Transformer Block × 4   │
│                         │
│ Multi-Head Attention    │
│ Layer Normalization     │
│ Feed Forward Network    │
│ Residual Connections    │
└─────────────────────────┘
   │
   ▼
Layer Normalization
   │
   ▼
Mean Pooling
   │
   ▼
Linear Classification Head
   │
   ▼
2 Logits
   │
   ├── Negative
   └── Positive
```

---

## Dataset

The project uses the Large Movie Review Dataset (IMDB).

- Training reviews: 25,000
- Testing reviews: 25,000
- Positive and negative classes are balanced
- 0 = Negative
- 1 = Positive

The original training dataset is shuffled and split into:

```text
Training:   22,500
Validation:  2,500
Testing:    25,000
```

The dataset itself is not stored in this repository.

---

## Data Processing

Tokenization uses:

```text
bert-base-uncased
```

Configuration:

```text
Maximum sequence length: 128
Batch size:              32
```

The custom `IMDBDataset` implements:

```python
__init__()
__len__()
__getitem__()
```

and integrates directly with PyTorch `DataLoader`.

---

## Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary | BERT tokenizer vocabulary |
| Embedding Dimension | 128 |
| Context Length | 128 |
| Transformer Layers | 4 |
| Attention Heads | 4 |
| Head Dimension | 32 |
| Dropout | 0.1 |
| Classes | 2 |

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Loss Function | CrossEntropyLoss |

---

## Results

Validation performance improved consistently during training:

| Epoch | Validation Accuracy |
|---:|---:|
| 1 | 73.24% |
| 2 | 76.96% |
| 3 | **79.16%** |

Final test performance:

```text
Test Accuracy: 76.49%
```

The final model exceeded the required **75% test accuracy** threshold.

---

## Training Behavior

Training loss decreased from approximately:

```text
0.688 → 0.413
```

while validation accuracy increased from:

```text
73.24% → 79.16%
```

This indicates that the model successfully learned useful sentiment representations from the IMDB review data.

---

## Project Structure

```text
sentimentscope-transformer-ml-pipeline/
│
├── notebooks/
│   └── SentimentScope.ipynb
│
├── models/
│   └── sentimentscope_model.pt
│
├── results/
├── src/
│
├── requirements.txt
├── .gitignore
└── README.md
```

The trained checkpoint is excluded from normal Git tracking because it exceeds GitHub's standard file-size limit.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- BERT Tokenizer
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook
- Git / GitHub

---

## Key Engineering Components

- Custom text dataset implementation
- Batch processing with PyTorch DataLoader
- Subword tokenization
- Multi-head self-attention
- Transformer blocks
- Residual connections
- Layer normalization
- Mean pooling
- Binary classification head
- Cross-entropy optimization
- Validation monitoring
- Model checkpointing
- Test-set evaluation

---

## Key Takeaways

1. A custom Transformer architecture can learn meaningful sentiment representations from large-scale movie review data.

2. Validation accuracy improved consistently across training epochs while training loss decreased.

3. BERT subword tokenization provides an efficient text representation while allowing the Transformer architecture itself to be implemented independently.

4. Model performance could be improved further through hyperparameter tuning, deeper Transformer architectures, larger embeddings, longer training, or more advanced pooling strategies.

---

## Future Improvements

Potential next steps include:

- Hyperparameter optimization
- Learning-rate scheduling
- Early stopping
- Attention-mask-aware pooling
- Bidirectional attention
- Model checkpoint selection based on validation accuracy
- Inference API
- Containerized deployment
- AWS-based model serving
- Experiment tracking and monitoring

---

## Future Enterprise Architecture

```text
Client Application
       │
       ▼
Amazon API Gateway
       │
       ▼
AWS Lambda / Container Service
       │
       ▼
SentimentScope Inference Service
       │
       ├── Model Artifact → Amazon S3
       ├── Logs → Amazon CloudWatch
       └── Metrics → Monitoring Dashboard
```

This architecture represents a potential future production evolution of the current local ML implementation.

---

## Attribution

This project was developed as part of Udacity's Transformer programming coursework.

The project dataset is based on the Large Movie Review Dataset introduced by Maas et al. (ACL 2011).

Starter project structure and educational guidance were provided by Udacity. My implementation work includes dataset loading, exploratory analysis, PyTorch Dataset/DataLoader integration, Transformer classification architecture, accuracy calculation, training, testing, evaluation, and model checkpoint creation.

---

## Author

**Natthida Sirapongkulpoj**

AI/ML Engineering Portfolio  
Python · PyTorch · Transformers · NLP · Computer Vision · AWS AI

