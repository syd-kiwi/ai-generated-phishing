from pathlib import Path
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def clean_text(t: str) -> str:
    t = str(t)
    t = t.lower()

    # remove urls and common email header tokens
    t = re.sub(r"(hxxps?|https?)://\S+", " ", t)
    t = re.sub(r"\bwww\.\S+", " ", t)
    t = re.sub(r"\bheader\b:", " ", t)

    # remove punctuation except apostrophes inside words
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    parser = argparse.ArgumentParser(description="Generate topic word clouds.")
    parser.add_argument(
        "--source",
        choices=["enron", "phishing"],
        default="enron",
        help="Dataset source to render word clouds for (default: enron).",
    )
    parser.add_argument("--emails-path", default=None, help="Path to emails/features table (parquet/csv).")
    parser.add_argument("--topics-path", default=None, help="Path to topic assignments CSV.")
    parser.add_argument("--output-dir", default=None, help="Directory to write word cloud image.")
    parser.add_argument("--topic-col", default=None, help="Topic column name (e.g., topic_id, dominant_topic).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    is_enron = args.source == "enron"
    if is_enron:
        emails_path = Path(args.emails_path) if args.emails_path else (
            project_root / "outputs" / "enron" / "message_level_features.csv"
        )
        topics_path = Path(args.topics_path) if args.topics_path else (
            project_root / "outputs" / "enron" / "topic_assignments.csv"
        )
    else:
        emails_path = Path(args.emails_path) if args.emails_path else (project_root / "outputs" / "emails.parquet")
        topics_path = Path(args.topics_path) if args.topics_path else (project_root / "outputs" / "topics.csv")

    out_dir = Path(args.output_dir) if args.output_dir else (project_root / "outputs" / ("enron/wordclouds" if is_enron else "wordclouds"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emails_path.exists():
        raise FileNotFoundError(f"Missing {emails_path}")
    if not topics_path.exists():
        raise FileNotFoundError(f"Missing {topics_path}. Run src/03_topics.py first.")

    emails = pd.read_parquet(emails_path) if emails_path.suffix == ".parquet" else pd.read_csv(emails_path)
    topics = pd.read_csv(topics_path)

    if is_enron:
        join_keys = [k for k in ["message_id", "label"] if k in emails.columns and k in topics.columns]
        if "combined_text" in emails.columns:
            emails["text"] = emails["combined_text"].fillna("").astype(str)
        else:
            emails["text"] = emails.get("subject", "").fillna("").astype(str) + " " + emails.get("message", "").fillna("").astype(str)
        topic_col = args.topic_col or "dominant_topic"
    else:
        join_keys = ["email_id", "label"]
        emails["text"] = emails["subject"].fillna("").astype(str) + " " + emails["raw_text"].fillna("").astype(str)
        topic_col = args.topic_col or "topic_id"

    if topic_col not in topics.columns:
        raise ValueError(f"Topic column '{topic_col}' not found in {topics_path}.")

    df = emails.merge(topics, on=join_keys, how="inner")
    df["topic_id"] = df[topic_col]

    # combine subject and body so urgency phrases in headers are included
    df["text"] = df["text"].apply(clean_text)

    # add custom stopwords to remove template filler words
    custom_stop = set(STOPWORDS)
    custom_stop |= {
        "dear","sincerely","regards","best","thank","thanks",
        "please","kindly","contact","team","support",
        "account","accounts","information","details","service","services",
        "required","request","requested","update","verification",
        "click","link","access","maintain","maintenance"
    }

    import math
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # collect topics first
    topic_groups = list(df.groupby("topic_id"))
    n_topics = len(topic_groups)

    # choose grid size
    ncols = 3
    nrows = math.ceil(n_topics / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax, (topic_id, sub) in zip(axes, topic_groups):
        text_blob = " ".join(sub["text"].dropna().astype(str).tolist()).strip()

        if not text_blob:
            ax.axis("off")
            ax.set_title(f"Topic {topic_id}\n(no text)")
            continue

        wc = WordCloud(
            width=1400,
            height=900,
            background_color="white",
            stopwords=custom_stop,
            collocations=True
        ).generate(text_blob)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {topic_id}", fontsize=12)

    # turn off any unused subplot boxes
    for ax in axes[n_topics:]:
        ax.axis("off")

    plt.tight_layout()

    out_path = out_dir / "all_topics_wordclouds.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Wrote", out_path)

if __name__ == "__main__":
    main()
