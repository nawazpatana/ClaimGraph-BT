#  MedClaimExtractor 

End-to-end implementation of the pipeline described in `complete_research_implementation.md`.

## Contents

| File | Purpose |
|---|---|
| `t1_claim_detection.ipynb` | SciBERT-based sentence-level claim/non-claim classifier with paper-level split, weighted loss, threshold tuning. |
| `t2_structured_extraction.ipynb` | Flan-T5 sequence-to-sequence extractor for the 7 structured fields. |
| `t3_claim_comparison.ipynb` | Cross-paper structured + semantic similarity |
| `t4_claim_relationship_graph.ipynb` | Builds semantic, performance, evolutionary,  |


## Recommended folder layout

Place the notebooks at the project root and create the directories below:

```
Research/
├── Data/
│   ├── Dataset.jsonl       # LLM-labelled training set (e.g. dataset_corrected_v2.jsonl renamed)
│   └── golden.jsonl        # manually labelled gold test set (e.g. golden_corrected_v2.jsonl renamed)
├── notebooks/
│   ├── t1_claim_detection.ipynb
│   ├── t2_structured_extraction.ipynb
│   ├── t3_claim_comparison.ipynb
│   └── t4_claim_relationship_graph.ipynb
├── outputs/                # created automatically; will hold all intermediate artifacts
├── models/                 # trained model checkpoints
└── dashboard_claim_mining.py
```

## Execution order

```
1. T1 → outputs/t1_claim_predictions.jsonl
2. T2 → outputs/t2_structured_claims.jsonl    (consumes T1)
3. T3 → outputs/t3_*.csv, t3_claims.parquet, t3_embeddings.npz   (consumes T2)
4. T4 → outputs/{semantic,performance,evolutionary,contradiction}_{edges.csv,graph.graphml}   (consumes T3)
5. Dashboard:  streamlit run dashboard_claim_mining.py
```

