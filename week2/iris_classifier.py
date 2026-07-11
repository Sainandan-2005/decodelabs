"""
Project 2: Data Classification Using AI
DecodeLabs - Industrial Training Kit (Batch 2026)

Goal: Build a basic classification model using a small dataset (Iris).
Pipeline (IPO Framework):
  INPUT   -> Load Iris dataset, scale features
  PROCESS -> Train-test split, KNN algorithm, tune K
  OUTPUT  -> Confusion matrix, F1 score, accuracy report

Key skills demonstrated: data handling, supervised learning basics, model training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# STEP 1: INPUT -- Load and understand the dataset
# ---------------------------------------------------------------------------
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("=" * 60)
    print("STEP 1: INPUT - The Iris Benchmark")
    print("=" * 60)
    print(f"Samples   : {df.shape[0]} (balanced)")
    print(f"Classes   : {df['species'].nunique()} -> {list(iris.target_names)}")
    print(f"Dimensions: {len(iris.feature_names)} -> {iris.feature_names}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass balance:")
    print(df["species"].value_counts())

    return iris.data, iris.target, iris.feature_names, iris.target_names, df


# ---------------------------------------------------------------------------
# STEP 2: PROCESS -- Scale, split, train
# ---------------------------------------------------------------------------
def scale_features(X):
    """The Gatekeeper Rule: StandardScaler -> mean=0, variance=1."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def split_data(X, y):
    """Structural Integrity: shuffle + 80/20 train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )
    print("\n" + "=" * 60)
    print("STEP 2: PROCESS - The Split")
    print("=" * 60)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set    : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def find_best_k(X_train, y_train, X_test, y_test, k_range=range(1, 31)):
    """Tuning the Engine: sweep K, find the elbow (lowest error rate).

    Note: K=1 almost always looks "best" on a single test split because it
    memorizes noise (overfitting - see slide 9). We deliberately search for
    the elbow starting at K=3 so the model generalizes instead of memorizing.
    """
    errors = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        errors.append(np.mean(preds != y_test))

    k_list = list(k_range)
    # search for elbow among K >= 3 to avoid the K=1 overfitting trap
    candidates = [(k, e) for k, e in zip(k_list, errors) if k >= 3]
    best_k, best_err = min(candidates, key=lambda t: t[1])

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), errors, marker="o", color="#1c3d5a")
    plt.axvline(best_k, color="orange", linestyle="--", label=f"Optimal K = {best_k}")
    plt.title("Tuning the Engine: Choosing K")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/iris_project/elbow_curve.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("STEP 2b: PROCESS - Tuning K (The Elbow)")
    print("=" * 60)
    print(f"Best K found: {best_k} (test error rate = {best_err:.3f}, K=1 excluded to avoid overfitting)")
    return best_k


def train_model(X_train, y_train, k):
    """The Workflow: instantiate -> fit -> (predict later)."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# STEP 3: OUTPUT -- Predict, validate, report
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print("\n" + "=" * 60)
    print("STEP 3: OUTPUT - Validation")
    print("=" * 60)
    print(f"Accuracy : {acc:.3f}")
    print(f"F1 Score : {f1:.3f} (macro-averaged)")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("The Diagnostic Tool: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("/home/claude/iris_project/confusion_matrix.png", dpi=150)
    plt.close()

    return acc, f1, predictions


def plot_decision_boundary(X, y, feature_names, target_names, k):
    """Bonus visual: decision boundary using the first 2 features only."""
    X2 = X[:, :2]
    scaler = StandardScaler()
    X2_scaled = scaler.fit_transform(X2)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X2_scaled, y)

    x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
    y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
    scatter = plt.scatter(
        X2_scaled[:, 0], X2_scaled[:, 1], c=y, cmap="viridis", edgecolor="k", s=40
    )
    plt.title(f"Decision Boundary (K={k}, first 2 features)")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(
        handles=scatter.legend_elements()[0], labels=list(target_names), title="Species"
    )
    plt.tight_layout()
    plt.savefig("/home/claude/iris_project/decision_boundary.png", dpi=150)
    plt.close()


def predict_new_sample(model, scaler, feature_names, target_names):
    """Simulate deploying the model on a brand-new, unseen flower."""
    new_sample = np.array([[5.9, 3.0, 5.1, 1.8]])  # looks like a virginica
    new_scaled = scaler.transform(new_sample)
    pred = model.predict(new_scaled)[0]
    proba = model.predict_proba(new_scaled)[0]

    print("\n" + "=" * 60)
    print("BONUS: Predicting a brand-new sample")
    print("=" * 60)
    print(f"Input features ({', '.join(feature_names)}): {new_sample[0]}")
    print(f"Predicted species: {target_names[pred]}")
    print("Class probabilities:")
    for name, p in zip(target_names, proba):
        print(f"  {name:12s}: {p:.2f}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    X, y, feature_names, target_names, df = load_data()

    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    best_k = find_best_k(X_train, y_train, X_test, y_test)
    model = train_model(X_train, y_train, best_k)

    acc, f1, predictions = evaluate_model(model, X_test, y_test, target_names)
    plot_decision_boundary(X, y, feature_names, target_names, best_k)
    predict_new_sample(model, scaler, feature_names, target_names)

    print("\n" + "=" * 60)
    print("PROJECT 2 COMPLETE ✅")
    print("=" * 60)
    print(f"Final Accuracy: {acc:.1%} | Final F1 Score: {f1:.3f}")
    print("Saved plots: elbow_curve.png, confusion_matrix.png, decision_boundary.png")


if __name__ == "__main__":
    main()
