---
library_name: transformers
pipeline_tag: text-classification
license: mit
language:
- en
tags:
- scientific-claim-detection
- biomedical-nlp
- brain-tumor
- scibert
- pubmedbert
- biobert
- modernbert
- text-classification
---

# T1 Claim Detection Model for Brain Tumor Research Abstracts

This repository contains the best T1 claim-detection model selected from a multi-model experiment for the thesis research project:

**Automated Scientific Claim Extraction, Comparison, and Knowledge Graph Construction from Brain Tumor Detection Literature Using Transformer-Based Models**.

The model classifies abstract sentences from brain tumor research papers into:

- `0`: non-claim
- `1`: claim

A claim is defined as a sentence that reports a result, finding, performance value, treatment effect, association, comparison, conclusion, recommendation, or evidence-based scientific statement.

## Best Model

- **Run name:** `BioBERT_Context`
- **Base checkpoint:** `monologg/biobert_v1.1_pubmed`
- **Selection metric:** `test_macro_f1`
- **Decision threshold:** `0.6500`

## Final Performance on Unseen Golden Test Set

The test set was manually annotated and kept unseen during training and threshold tuning.

| Metric | Score |
|---|---:|
| Accuracy | 0.8393 |
| Macro F1 | 0.8356 |
| Claim Precision | 0.8304 |
| Claim Recall | 0.7924 |
| Claim F1 | 0.8110 |
| AUC | 0.9265 |

### Confusion Matrix

Rows are true labels and columns are predicted labels.

|  | Pred non-claim | Pred claim |
|---|---:|---:|
| True non-claim | 893 | 127 |
| True claim | 163 | 622 |

## Dataset and Evaluation Design

- Domain: brain tumor research abstracts
- Total corpus: 1,586 papers
- Golden/manual subset: 316 papers
- Training: non-golden LLM-labelled papers
- Validation: manually annotated golden validation papers
- Test: unseen manually annotated golden test papers
- Split level: paper-level split to avoid DOI/paper leakage
- Threshold tuning: performed only on golden validation, not on golden test

## Multi-Model Comparison

| run_name           |   test_accuracy |   test_macro_f1 |   test_claim_f1 |   test_claim_precision |   test_claim_recall |   test_auc |   best_threshold |
|:-------------------|----------------:|----------------:|----------------:|-----------------------:|--------------------:|-----------:|-----------------:|
| BioBERT_Context    |          0.8393 |          0.8356 |          0.811  |                 0.8304 |              0.7924 |     0.9265 |             0.65 |
| SciBERT_Context    |          0.831  |          0.8274 |          0.8026 |                 0.8158 |              0.7898 |     0.9182 |             0.17 |
| PubMedBERT_Context |          0.826  |          0.817  |          0.7764 |                 0.8805 |              0.6943 |     0.9248 |             0.56 |
| ModernBERT_Context |        nan      |        nan      |        nan      |               nan      |            nan      |   nan      |           nan    |

## Intended Use

This model is intended for research use in a scientific claim-mining pipeline. It can be used as the first stage for identifying claim sentences before structured extraction of fields such as model, task, dataset, metric, value, comparison, and domain.

## Limitations

- The model was trained for brain tumor research abstracts and may not generalize directly to unrelated biomedical domains.
- It may confuse methods/objectives with claims when a sentence is written in a result-like style.
- It may miss implicit claims that do not contain clear result or conclusion signals.
- It should not be used as a clinical decision-making tool.

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo_id = "YOUR_USERNAME/YOUR_REPO_NAME"
threshold = 0.6500

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
model.eval()

sentence = "The proposed model achieved a Dice score of 0.91 on the BraTS 2020 dataset."
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=256)

with torch.no_grad():
    logits = model(**inputs).logits
    prob_claim = torch.softmax(logits, dim=-1)[0, 1].item()

prediction = int(prob_claim >= threshold)
print({"prob_claim": prob_claim, "prediction": prediction})
```

## Training Details

- Architecture: `AutoModelForSequenceClassification`
- Loss: weighted cross-entropy
- Positive claim class weight: `3.0`
- Context input: enabled
- Optimizer: AdamW
- Learning rate: `2e-05`
- Max sequence length: `256`
- Epochs per model: `4`
- Export date: `2026-05-12T02:55:39.937235`
