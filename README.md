# Banking77 Intent Classification

Benchmarking three approaches to intent classification on the [Banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset, which contains 13,083 customer service queries across 77 banking intents.

## Models

| Method | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 89.03% | 89.06% |
| DistilBERT (fine-tuned) | 92.37% | 92.36% |
| BART-large (zero-shot) | 35.45% | 34.52% |

## Structure

```
banking77_main.ipynb   # main notebook
requirements.txt
```

The notebook covers:
- EDA on the Banking77 dataset
- TF-IDF + Logistic Regression baseline
- DistilBERT fine-tuning with the HuggingFace Trainer API
- Zero-shot classification with `facebook/bart-large-mnli`
- Per-class evaluation and confusion matrix

## Setup

```bash
pip install -r requirements.txt
```

A GPU is recommended for the DistilBERT fine-tuning step. The zero-shot BART section takes around 77 minutes on a single GPU due to the number of candidate labels.

## Dataset

[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) via HuggingFace Datasets. 10,003 training examples and 3,080 test examples, 40 per intent.
