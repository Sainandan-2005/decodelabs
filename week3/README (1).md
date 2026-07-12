# Tech Stack Recommender

A content-based recommendation engine built for **DecodeLabs Project 3: AI Recommendation Logic**.

It maps a user's raw skills to the most relevant job roles using **TF-IDF weighting** and **Cosine Similarity** — no machine-learning libraries required, just the Python standard library.

## How it works

The engine follows a strict Input → Process → Output pipeline:

1. **Ingestion** — capture at least 3 user skills
2. **Vector Mapping** — build a shared vocabulary from every job role's tags plus the user's tags
3. **TF-IDF Weighting** — score each tag by how often it appears for an item (Term Frequency) and how rare it is across all items (Inverse Document Frequency), so generic skills like "automation" count for less than specific ones
4. **Scoring** — compute cosine similarity between the user's vector and every job role's vector (measures angle/orientation, not raw magnitude)
5. **Sorting** — rank job roles by descending similarity score
6. **Filtering** — return only the Top 3 matches, avoiding choice overload

This is a **content-based** filtering approach (item attributes only), so it works immediately with no historical user data — it's inherently robust against the cold-start problem for new items.

## Files

| File | Purpose |
|---|---|
| `recommender.py` | The recommendation engine + CLI |
| `raw_skills.csv` | Dataset of job roles and their associated skills (the "items") |
| `README.md` | This file |

## Requirements

- Python 3.7 or later
- No external packages — uses only `csv`, `math`, `sys`, and `collections` from the standard library

## Setup

1. Download `recommender.py` and `raw_skills.csv` into the **same folder**.
2. Open a terminal in that folder.

## Usage

### Option A — Interactive mode

```bash
python3 recommender.py
```

You'll be prompted:

```
Enter at least 3 skills or interests, separated by commas.
Example: Python, Cloud Computing, Automation
> Python, Cloud Computing, Automation
```

### Option B — Command-line arguments

```bash
python3 recommender.py "Python" "Cloud Computing" "Automation"
```

### Example output

```
Input skills: ['Python', 'Cloud Computing', 'Automation']
Top 3 recommended career paths:

  1. Cloud Architect          match:  43.2%   skills: aws, azure, cloud computing, networking, automation, terraform
  2. Security Engineer        match:  33.8%   skills: networking, linux, security, automation, python
  3. Database Administrator   match:  15.7%   skills: sql, database design, backup, automation, linux
```

> Note: if your system uses `python` instead of `python3`, swap the command accordingly.

## Customizing the dataset

`raw_skills.csv` uses a simple two-column format:

```csv
job_role,skills
Data Scientist,"Python,SQL,Machine Learning,Statistics,Data Analysis,Pandas"
```

To add or edit job roles, add or edit rows in this format — no code changes are needed. The vocabulary and TF-IDF weights are rebuilt automatically every time the script runs.

## Requirement enforced in code

At least 3 skills must be provided (per the project's Ingestion requirement). Fewer than 3 raises a clear error instead of producing an unreliable match.

## Limitations

- **Cold start**: if a user's skills share zero terms with any job role in the dataset, all scores will be 0.0 (there's nothing to match against yet).
- **Vocabulary-dependent**: skill names must roughly match the wording used in `raw_skills.csv` (e.g. "Cloud Computing" vs "Cloud Comp" would be treated as different terms) — this is the "language barrier" the TF-IDF/vector-mapping step is designed to solve, but only within a consistent vocabulary.
