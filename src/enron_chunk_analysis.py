from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def count_sentences(text: str) -> int:
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return len(parts)


def lexical_diversity(text: str) -> float:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def average_word_length(text: str) -> float:
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1

    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel

    if w.endswith("e") and syllables > 1:
        syllables -= 1

    return max(syllables, 1)


def fallback_flesch_kincaid_grade(text: str) -> float:
    words = WORD_RE.findall(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0

    n_sentences = max(count_sentences(text), 1)
    n_syllables = sum(estimate_syllables(w) for w in words)
    return 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59


def safe_flesch_kincaid_grade(text: str) -> float:
    try:
        nltk.data.find("corpora/cmudict")
        return float(textstat.flesch_kincaid_grade(text))
    except LookupError:
        return float(fallback_flesch_kincaid_grade(text))


def normalize_label(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"spam", "1", "true"}:
        return "spam"
    return "ham"


def load_enron_csv(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    col_map = {c.lower().strip(): c for c in df.columns}
    required = ["message id", "subject", "message", "spam/ham", "date"]
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}. Expected columns: Message ID,Subject,Message,Spam/Ham,Date"
        )

    out = pd.DataFrame(
        {
            "message_id": df[col_map["message id"]].astype(str),
            "subject": df[col_map["subject"]].fillna("").astype(str),
            "message": df[col_map["message"]].fillna("").astype(str),
            "label": df[col_map["spam/ham"]].apply(normalize_label),
            "date": df[col_map["date"]].astype(str),
        }
    )
    out["combined_text"] = out["subject"] + "\n" + out["message"]
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["word_count"] = out["combined_text"].apply(count_words)
    out["sentence_count"] = out["combined_text"].apply(count_sentences)
    out["char_count"] = out["combined_text"].str.len()
    out["avg_word_length"] = out["combined_text"].apply(average_word_length)
    out["lexical_diversity"] = out["combined_text"].apply(lexical_diversity)
    out["flesch_kincaid_grade"] = out["combined_text"].apply(safe_flesch_kincaid_grade)

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        sia = SentimentIntensityAnalyzer()
        sentiment_df = out["combined_text"].apply(sia.polarity_scores).apply(pd.Series)
    except LookupError:
        sentiment_df = pd.DataFrame(
            [{"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}] * len(out)
        )

    out = pd.concat([out, sentiment_df], axis=1)
    return out


def make_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "word_count",
        "sentence_count",
        "char_count",
        "avg_word_length",
        "lexical_diversity",
        "flesch_kincaid_grade",
        "neg",
        "neu",
        "pos",
        "compound",
    ]
    return df.groupby("label")[numeric_cols].agg(["count", "mean", "median", "std"]).reset_index()


def top_terms_by_label(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby("label"):
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
        X = vectorizer.fit_transform(group["combined_text"])
        counts = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vectorizer.get_feature_names_out())
        top_idx = np.argsort(counts)[::-1][:top_n]
        for i in top_idx:
            rows.append({"label": label, "term": vocab[i], "count": int(counts[i])})
    return pd.DataFrame(rows)


def lda_topics(df: pd.DataFrame, n_components: int = 6, top_words: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        return pd.DataFrame(), pd.DataFrame()

    n_components = max(2, min(n_components, len(df) - 1))

    vectorizer = CountVectorizer(stop_words="english", max_features=2500)
    X = vectorizer.fit_transform(df["combined_text"])

    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    doc_topic = lda.fit_transform(X)

    topic_assignments = pd.DataFrame(
        {
            "message_id": df["message_id"],
            "label": df["label"],
            "dominant_topic": np.argmax(doc_topic, axis=1),
        }
    )

    vocab = np.array(vectorizer.get_feature_names_out())
    topic_rows = []
    for topic_id, comp in enumerate(lda.components_):
        idx = np.argsort(comp)[::-1][:top_words]
        topic_rows.append(
            {
                "topic": topic_id,
                "top_words": ", ".join(vocab[idx]),
            }
        )
    topic_words = pd.DataFrame(topic_rows)
    return topic_assignments, topic_words


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def model_comparison(df: pd.DataFrame) -> dict:
    label_counts = df["label"].value_counts()
    if df["label"].nunique() < 2:
        raise ValueError("Need both spam and ham labels to train classifiers.")
    if int(label_counts.min()) < 2:
        return {
            "status": "not_trained_too_few_examples_per_class",
            "label_counts": label_counts.to_dict(),
            "note": "Need at least 2 messages per class.",
        }

    feature_cols = [
        "word_count",
        "sentence_count",
        "char_count",
        "avg_word_length",
        "lexical_diversity",
        "flesch_kincaid_grade",
        "neg",
        "neu",
        "pos",
        "compound",
    ]

    X = df[["combined_text", *feature_cols]]
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2 if len(df) >= 25 else 0.33,
        random_state=42,
        stratify=y,
    )

    prep = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=10000), "combined_text"),
            ("num", Pipeline([("scale", StandardScaler(with_mean=False))]), feature_cols),
        ]
    )

    baseline = Pipeline(
        steps=[
            ("prep", prep),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]

    results = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_counts": label_counts.to_dict(),
        "baseline_logistic_regression": compute_metrics(y_test, baseline_pred, baseline_prob),
    }

    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        backend = "xgboost"
    except ModuleNotFoundError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        xgb_model = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=8, random_state=42)
        backend = "hist_gradient_boosting_fallback"

    xgb_pipe = Pipeline(steps=[("prep", prep), ("model", xgb_model)])
    xgb_pipe.fit(X_train, y_train)
    xgb_pred = xgb_pipe.predict(X_test)
    xgb_prob = xgb_pipe.predict_proba(X_test)[:, 1]

    results["xgboost_backend"] = backend
    results["xgboost_or_fallback"] = compute_metrics(y_test, xgb_pred, xgb_prob)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze spam vs ham from CSV schema Message ID,Subject,Message,Spam/Ham,Date and compare baseline vs XGBoost."
        )
    )
    parser.add_argument("--input", required=True, help="CSV file path with columns Message ID,Subject,Message,Spam/Ham,Date")
    parser.add_argument("--output-dir", default="outputs/enron", help="Output directory for reports.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_enron_csv(input_path)
    featured = build_feature_frame(df)

    summary = make_label_summary(featured)
    terms = top_terms_by_label(featured, top_n=20)
    topic_assignments, topic_words = lda_topics(featured)
    models = model_comparison(featured)

    featured.to_csv(output_dir / "message_level_features.csv", index=False)
    summary.to_csv(output_dir / "spam_ham_summary.csv", index=False)
    terms.to_csv(output_dir / "top_terms_by_label.csv", index=False)
    if not topic_assignments.empty:
        topic_assignments.to_csv(output_dir / "topic_assignments.csv", index=False)
        topic_words.to_csv(output_dir / "topic_words.csv", index=False)
    (output_dir / "model_comparison.json").write_text(json.dumps(models, indent=2), encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Rows: {len(featured)}")
    print("Label counts:")
    print(featured["label"].value_counts())
    print(f"Wrote: {output_dir / 'message_level_features.csv'}")
    print(f"Wrote: {output_dir / 'spam_ham_summary.csv'}")
    print(f"Wrote: {output_dir / 'top_terms_by_label.csv'}")
    if not topic_assignments.empty:
        print(f"Wrote: {output_dir / 'topic_assignments.csv'}")
        print(f"Wrote: {output_dir / 'topic_words.csv'}")
    print(f"Wrote: {output_dir / 'model_comparison.json'}")
    print("Model comparison:")
    print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()