# AI-Generated Phishing Email Analysis

This repository contains a small NLP analysis pipeline for exploring a corpus of AI-generated phishing emails.

The project:
- ingests raw `.txt` phishing emails,
- extracts linguistic and social-engineering features,
- runs topic modeling and sentiment scoring,
- computes readability metrics,
- generates visualizations and topic word clouds.

## Repository structure

- `data/` — raw phishing email text files (`email_*.txt`).
- `src/` — analysis scripts.
- `outputs/` — generated datasets, figures, and word cloud images.

## Requirements

Install Python dependencies:

```bash
pip install -r src/requirements.txt
```

> Note: `nltk` VADER lexicon is downloaded automatically by `src/email_sentiment.py`.

## Pipeline

Run scripts from the repository root in this order:

```bash
python src/data_processing.py
python src/email_feature_extraction.py
python src/email_topics.py
python src/email_sentiment.py
python src/email_readability.py
python src/visualizations.py
python src/word_clouds.py
```

## Script details

### `src/data_processing.py`
- Reads all `.txt` files in `data/`.
- Extracts `Header:` as an email subject when present.
- Produces `outputs/emails.parquet`.

### `src/email_feature_extraction.py`
Builds rule-based phishing indicators, including:
- URL and obfuscation detection,
- urgency/fear/authority/action/account/reward trigger scores,
- token and phrase hit rates per 100 words.

Output: `outputs/features.csv`.

### `src/email_topics.py`
- Uses `CountVectorizer` + LDA (`n_components=8`) for topic modeling.
- Outputs per-email topic assignments and per-topic top words.

Outputs:
- `outputs/topics.csv`
- `outputs/topic_words.csv`

### `src/email_sentiment.py`
- Uses NLTK VADER sentiment scoring on subject + body text.
- Exports `neg`, `neu`, `pos`, and `compound` sentiment features.

Output: `outputs/sentiment.csv`.

### `src/email_readability.py`
Computes readability and lexical metrics:
- Flesch-Kincaid grade,
- average sentence length,
- lexical diversity.

Output: `outputs/readability.csv`.

### `src/visualizations.py`
Generates figure outputs under `outputs/figures/`, including:
- sentiment/readability histograms,
- metric summary boxplot,
- top words bar chart (+ CSV counts),
- top bigrams bar chart (+ CSV counts).

### `src/word_clouds.py`
- Merges topic assignments with email content.
- Builds one word cloud per discovered topic.

Output directory: `outputs/wordclouds/`.

## Outputs produced

This repo already includes generated artifacts, such as:
- `outputs/emails.parquet`
- `outputs/features.csv`
- `outputs/topics.csv`, `outputs/topic_words.csv`
- `outputs/sentiment.csv`
- `outputs/readability.csv`
- `outputs/figures/*.png`, `outputs/figures/*_counts.csv`
- `outputs/wordclouds/topic_*_wordcloud.png`

## Typical workflow

1. Add or replace raw emails in `data/`.
2. Re-run the pipeline commands above.
3. Inspect CSV outputs for modeling/analysis.
4. Use generated plots and word clouds for reporting.
