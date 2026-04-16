from pathlib import Path
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Helpers
# ----------------------------
URL_RE = re.compile(r"(?i)\b(?:hxxps?|https?)://\S+|\bwww\.\S+")

def clean_for_tokens(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text)

def main():
    parser = argparse.ArgumentParser(description="Generate summary plots for phishing/email datasets.")
    parser.add_argument("--emails-path", default=None, help="Path to emails table (parquet/csv).")
    parser.add_argument("--sentiment-path", default=None, help="Path to sentiment CSV.")
    parser.add_argument("--readability-path", default=None, help="Path to readability CSV.")
    parser.add_argument("--output-dir", default=None, help="Directory for output figures.")
    parser.add_argument(
        "--enron",
        action="store_true",
        help="Use Enron outputs (outputs/enron/message_level_features.csv) as a single input.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else (
        project_root / "outputs" / ("enron/figures" if args.enron else "figures")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    if args.enron:
        enron_features_path = Path(args.readability_path) if args.readability_path else (
            project_root / "outputs" / "enron" / "message_level_features.csv"
        )
        if not enron_features_path.exists():
            raise FileNotFoundError(f"Missing {enron_features_path}. Run src/enron_chunk_analysis.py first.")
        df = pd.read_csv(enron_features_path)
        text_series = df.get("combined_text", "")
    else:
        # Inputs
        emails_path = Path(args.emails_path) if args.emails_path else (project_root / "outputs" / "emails.parquet")
        sentiment_path = Path(args.sentiment_path) if args.sentiment_path else (project_root / "outputs" / "sentiment.csv")
        readability_path = Path(args.readability_path) if args.readability_path else (project_root / "outputs" / "readability.csv")

        if not emails_path.exists():
            raise FileNotFoundError(f"Missing {emails_path}. Run ingest first.")
        if not sentiment_path.exists():
            raise FileNotFoundError(f"Missing {sentiment_path}. Run sentiment script first.")
        if not readability_path.exists():
            raise FileNotFoundError(f"Missing {readability_path}. Run readability script first.")

        emails = pd.read_parquet(emails_path) if emails_path.suffix == ".parquet" else pd.read_csv(emails_path)
        sent = pd.read_csv(sentiment_path)
        read = pd.read_csv(readability_path)

        # Merge for metric plots
        df = read.merge(sent[["email_id", "compound"]], on="email_id", how="inner")
        text_series = emails["subject"].fillna("").astype(str) + " " + emails["raw_text"].fillna("").astype(str)

    # ----------------------------
    # 1) Metric histograms (Seaborn)
    # ----------------------------
    def hist_plot(series, title, xlabel, filename, bins=30):
        plt.figure(figsize=(8, 5))
        sns.histplot(series.dropna(), bins=bins, color="#4C72B0")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=300)
        plt.close()

    hist_plot(df["compound"], "Sentiment Distribution (VADER Compound)", "Compound Score", "sentiment_distribution.png", bins=30)
    hist_plot(df["flesch_kincaid_grade"], "Flesch-Kincaid Grade Level Distribution", "Grade Level", "grade_level_distribution.png", bins=25)
    hist_plot(df["avg_sentence_length"], "Average Sentence Length Distribution", "Words per Sentence", "sentence_length_distribution.png", bins=25)
    hist_plot(df["lexical_diversity"], "Lexical Diversity Distribution", "Lexical Diversity", "lexical_diversity_distribution.png", bins=25)

    # ----------------------------
    # 2) Summary boxplot (Seaborn)
    # ----------------------------
    metrics = df[["compound", "flesch_kincaid_grade", "avg_sentence_length", "lexical_diversity"]].dropna()
    metrics_long = metrics.melt(var_name="metric", value_name="value")
    label_map = {
        "compound": "Sentiment",
        "flesch_kincaid_grade": "Grade",
        "avg_sentence_length": "Sent Len",
        "lexical_diversity": "Lex Div",
    }
    metrics_long["metric"] = metrics_long["metric"].map(label_map)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics_long, x="metric", y="value")
    plt.title("Distribution of Key Linguistic Metrics")
    plt.xlabel("")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_boxplot.png", dpi=300)
    plt.close()

    # ----------------------------
    # 3) Top words bar chart
    # ----------------------------
    combined_text = pd.Series(text_series).fillna("").astype(str)
    all_tokens = []
    for t in combined_text:
        cleaned = clean_for_tokens(t)
        all_tokens.extend(tokenize(cleaned))

    # Stopwords: remove common English + template filler.
    stop = {
        "the","a","an","and","or","but","if","to","of","in","on","for","with","as","at","by","from",
        "is","are","was","were","be","been","being","it","this","that","these","those","you","your",
        "we","our","us","they","their","them","i","me","my",

        # email filler
        "please","kindly","thank","thanks","regards","sincerely","dear","best",
        "contact","team","support",

        # high-frequency template words (remove so plot is meaningful)
        "account","accounts","information","details","update","verify","verification","security","click","link",
        "required","request","requested","ensure","maintain","maintenance","access","process"
    }

    tokens_filt = [t for t in all_tokens if t not in stop and len(t) > 2 and not t.isdigit()]
    counts = Counter(tokens_filt)

    top_n = 25
    most_common = counts.most_common(top_n)
    top_words_df = pd.DataFrame(most_common, columns=["word", "count"])
    top_words_df.to_csv(out_dir / "top_words_counts.csv", index=False)

    # plot horizontal bar
    top_words_plot = top_words_df.sort_values("count", ascending=True)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_words_plot, x="count", y="word", orient="h", color="#4C72B0")
    plt.title(f"Top {top_n} Most Common Words (Stopwords and Template Words Removed)")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "top_words_bar.png", dpi=300)
    plt.close()

    # ----------------------------
    # 4) Top bigrams bar chart
    # ----------------------------
    texts_for_vec = combined_text.astype(str).tolist()

    min_df = 5 if len(texts_for_vec) >= 5 else 1
    vec = CountVectorizer(
        stop_words="english",
        ngram_range=(2, 2),
        min_df=min_df
    )
    X = vec.fit_transform(texts_for_vec)
    freqs = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()

    top_bi = 20
    idx = freqs.argsort()[-top_bi:]
    top_terms = terms[idx]
    top_freqs = freqs[idx]

    # sort ascending for barh
    order = top_freqs.argsort()
    top_terms = top_terms[order]
    top_freqs = top_freqs[order]

    bigrams_df = pd.DataFrame({"bigram": top_terms, "count": top_freqs})
    bigrams_df.to_csv(out_dir / "top_bigrams_counts.csv", index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=bigrams_df, x="count", y="bigram", orient="h", color="#4C72B0")
    plt.title(f"Top {top_bi} Most Common Bigrams")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "top_bigrams_bar.png", dpi=300)
    plt.close()

    print("Saved figures to:", out_dir)
    print("Saved counts:")
    print(" -", out_dir / "top_words_counts.csv")
    print(" -", out_dir / "top_bigrams_counts.csv")

if __name__ == "__main__":
    main()
