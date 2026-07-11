# Project 2: Data Classification Using AI
**DecodeLabs Industrial Training Kit — Batch 2026**

A small supervised-learning project that classifies Iris flowers into
species using **K-Nearest Neighbors (KNN)**, following the IPO
(Input → Process → Output) framework from the training deck.

## What it does

| Stage | Steps |
|---|---|
| **Input** | Load the classic Iris dataset (150 samples, 3 classes, 4 features) and scale features with `StandardScaler` |
| **Process** | Shuffle + split into 80% train / 20% test, sweep K from 1–30 to find the error-rate "elbow" (excluding K=1, which overfits/memorizes noise), train the final `KNeighborsClassifier` |
| **Output** | Confusion matrix, classification report, accuracy, macro F1 score, decision boundary plot, and a live prediction on a brand-new unseen flower |

## Files

- `iris_classifier.py` — the full pipeline, runnable end to end
- `elbow_curve.png` — error rate vs K, showing the chosen optimal K
- `confusion_matrix.png` — TP/FP/FN/TN breakdown per species
- `decision_boundary.png` — visual decision regions using the first 2 features

## Run it

```bash
pip install scikit-learn matplotlib pandas numpy seaborn
python iris_classifier.py
```

## Result snapshot

- **Accuracy:** ~96.7%
- **Macro F1 Score:** ~0.97
- Best K found via elbow search (K=1 deliberately excluded to avoid overfitting)

## Why this matters (per the training kit)

- **Scaling first** — unscaled features bias distance-based algorithms like KNN
- **Shuffle before splitting** — removes order bias, keeps train/test truly random
- **Accuracy alone can lie** — always check the confusion matrix and F1 score, especially on imbalanced data
- **K=1 is a trap** — it memorizes training noise rather than learning the general pattern

## Next milestone

The deck points to **Deep Learning & CNNs** as the natural next step —
moving from tabular data (rows/columns) to image data (pixel grids).
