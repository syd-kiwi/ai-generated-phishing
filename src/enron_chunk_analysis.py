from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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


FILE_RE = re.compile(r"(?im)^\s*file\s*:\s*(.+?)\s*$")
MESSAGE_RE = re.compile(r"(?im)^\s*message\s*:\s*$")
LABEL_RE = re.compile(r"(?im)^\s*(?:label\s*:\s*)?(spam|ham)\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


@dataclass
class ChunkRecord:
    chunk_id: int
    source_file: str
    message: str
    label: str


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


def parse_chunks(raw_text: str) -> list[ChunkRecord]:
    file_matches = list(FILE_RE.finditer(raw_text))
    chunks: list[ChunkRecord] = []

    if not file_matches:
        return chunks

    for idx, match in enumerate(file_matches):
        start = match.start()
        end = file_matches[idx + 1].start() if idx + 1 < len(file_matches) else len(raw_text)
        block = raw_text[start:end].strip()

        source_file = match.group(1).strip()

        message_marker = MESSAGE_RE.search(block)
        if not message_marker:
            continue

        body_start = message_marker.end()
        post_message = block[body_start:].strip("\n")
        lines = post_message.splitlines()

        label = None
        label_line_idx = None
        for rev_idx, line in enumerate(reversed(lines)):
            m = LABEL_RE.match(line.strip())
            if m:
                label = m.group(1).lower()
                label_line_idx = len(lines) - 1 - rev_idx
                break

        if label is None:
            continue

        message_lines = lines[:label_line_idx]
        message = "\n".join(message_lines).strip()

        chunks.append(
            ChunkRecord(
                chunk_id=len(chunks),
                source_file=source_file,
                message=message,
                label=label,
            )
        )

    return chunks


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["word_count"] = out["message"].apply(count_words)
    out["sentence_count"] = out["message"].apply(count_sentences)
    out["char_count"] = out["message"].str.len()
    out["avg_word_length"] = out["message"].apply(average_word_length)
    out["lexical_diversity"] = out["message"].apply(lexical_diversity)
    out["flesch_kincaid_grade"] = out["message"].apply(safe_flesch_kincaid_grade)

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        sia = SentimentIntensityAnalyzer()
        sentiment_df = out["message"].apply(sia.polarity_scores).apply(pd.Series)
    except LookupError:
        sentiment_df = pd.DataFrame(
            [{"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}] * len(out)
        )

    out = pd.concat([out, sentiment_df], axis=1)
    return out


def train_xgboost_classifier(df: pd.DataFrame) -> dict:
    model_name = "xgboost"
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        XGBClassifier = None
        model_name = "hist_gradient_boosting_fallback"

    if df["label"].nunique() < 2:
        raise ValueError("Need both ham and spam samples to train XGBoost classifier.")

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

    X = df[["message", *feature_cols]]
    y = (df["label"] == "spam").astype(int)

    test_size = 0.2 if len(df) >= 25 else 0.33
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000), "message"),
            ("num", Pipeline([("scale", StandardScaler(with_mean=False))]), feature_cols),
        ],
    )

    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
        )

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model_used": model_name,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def resolve_input_file(path: Path) -> Path:
    if path.is_file():
        return path

    if path.is_dir():
        candidates = sorted(
            [
                p
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".log"}
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No candidate text-like files found under {path}")
        return candidates[0]

    raise FileNotFoundError(f"Input path does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse chunked Enron-like file/message/label text, analyze ham-only chunks, and train XGBoost spam classifier."
        )
    )
    parser.add_argument(
        "--input",
        default="data/enron",
        help="Path to a single file or a directory containing the chunked file. Defaults to data/enron.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/enron",
        help="Directory where outputs are written.",
    )

    args = parser.parse_args()

    input_path = resolve_input_file(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
    records = parse_chunks(raw_text)

    if not records:
        raise RuntimeError(
            "No valid chunks parsed. Expected repeated blocks with file:, message:, and final spam/ham label line."
        )

    parsed_df = pd.DataFrame([r.__dict__ for r in records])
    parsed_df = build_feature_frame(parsed_df)

    ham_df = parsed_df[parsed_df["label"] == "ham"].copy()

    parsed_path = output_dir / "parsed_chunks.csv"
    ham_path = output_dir / "ham_only_analysis.csv"
    metrics_path = output_dir / "xgboost_metrics.json"

    parsed_df.to_csv(parsed_path, index=False)
    ham_df.to_csv(ham_path, index=False)

    metrics = train_xgboost_classifier(parsed_df)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Input file: {input_path}")
    print(f"Total parsed chunks: {len(parsed_df)}")
    print("Label counts:")
    print(parsed_df["label"].value_counts())
    print(f"Ham chunks analyzed: {len(ham_df)}")
    print(f"Wrote: {parsed_path}")
    print(f"Wrote: {ham_path}")
    print(f"Wrote: {metrics_path}")
    print("XGBoost metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
