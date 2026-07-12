"""
Tech Stack Recommender — Project 3 Capstone (DecodeLabs)
=========================================================
A content-based filtering engine that maps a user's raw skills to the
most relevant job roles using TF-IDF weighting + Cosine Similarity.

Pipeline (per the IPO model):
  1. Ingestion   -> capture user skills (min 3)
  2. Scoring     -> TF-IDF vectorize + cosine similarity vs every job role
  3. Sorting     -> rank by descending similarity score
  4. Filtering   -> return Top-N matches

Run:
    python recommender.py
    python recommender.py "Python" "Cloud Computing" "Automation"
"""

import csv
import math
import sys
from collections import Counter


# --------------------------------------------------------------------------
# Step 0: Load the item catalog (job roles + their skill tags)
# --------------------------------------------------------------------------
def load_dataset(path="raw_skills.csv"):
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = [t.strip().lower() for t in row["skills"].split(",") if t.strip()]
            items.append({"name": row["job_role"], "tags": tags})
    return items


# --------------------------------------------------------------------------
# Step 1: Vector Mapping — build a shared vocabulary space
# --------------------------------------------------------------------------
def build_vocabulary(documents):
    """documents: list of tag-lists (each item's tags, plus the user's tags)"""
    vocab = sorted({tag for doc in documents for tag in doc})
    return {term: idx for idx, term in enumerate(vocab)}


# --------------------------------------------------------------------------
# Step 2: TF-IDF weighting (upgrades binary overlap into weighted vectors)
# --------------------------------------------------------------------------
def compute_idf(documents, vocab):
    n_docs = len(documents)
    idf = {}
    for term in vocab:
        docs_with_term = sum(1 for doc in documents if term in doc)
        # +1 smoothing avoids div-by-zero / log(0) for terms present everywhere
        idf[term] = math.log((n_docs + 1) / (docs_with_term + 1)) + 1
    return idf


def vectorize(doc, vocab, idf):
    vec = [0.0] * len(vocab)
    total_terms = len(doc) if doc else 1
    tf = Counter(doc)
    for term, count in tf.items():
        if term in vocab:
            tf_score = count / total_terms
            vec[vocab[term]] = tf_score * idf[term]
    return vec


# --------------------------------------------------------------------------
# Step 3: Cosine Similarity — the industry-standard similarity metric
# --------------------------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0  # Cold Start: a zero-vector cannot be scored
    return dot / (mag_a * mag_b)


# --------------------------------------------------------------------------
# The 4-step ranking pipeline: Ingestion -> Scoring -> Sorting -> Filtering
# --------------------------------------------------------------------------
def recommend(user_skills, items, top_n=3):
    if len(user_skills) < 3:
        raise ValueError("Please provide at least 3 skills (Ingestion requirement).")

    user_tags = [s.strip().lower() for s in user_skills]

    # Build a shared vocabulary across every item + the user profile
    all_docs = [item["tags"] for item in items] + [user_tags]
    vocab = build_vocabulary(all_docs)
    idf = compute_idf(all_docs, vocab)

    user_vector = vectorize(user_tags, vocab, idf)

    # Scoring: cosine similarity of user profile against every job role
    scored = []
    for item in items:
        item_vector = vectorize(item["tags"], vocab, idf)
        score = cosine_similarity(user_vector, item_vector)
        scored.append((item["name"], score, item["tags"]))

    # Sorting: descending by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Filtering: Top-N only (prevents choice overload)
    return scored[:top_n]


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def main():
    items = load_dataset("raw_skills.csv")

    if len(sys.argv) > 1:
        user_skills = sys.argv[1:]
    else:
        print("Enter at least 3 skills or interests, separated by commas.")
        print("Example: Python, Cloud Computing, Automation")
        raw = input("> ")
        user_skills = [s.strip() for s in raw.split(",") if s.strip()]

    try:
        results = recommend(user_skills, items, top_n=3)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\nInput skills: {user_skills}")
    print("Top 3 recommended career paths:\n")
    for rank, (name, score, tags) in enumerate(results, start=1):
        pct = round(score * 100, 1)
        print(f"  {rank}. {name:<24} match: {pct:>5}%   skills: {', '.join(tags)}")


if __name__ == "__main__":
    main()
