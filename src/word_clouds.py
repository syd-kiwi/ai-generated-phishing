from pathlib import Path
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import math

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
    parser = argparse.ArgumentParser(description="Generate topic word clouds from topic_assignments.csv + topic_words.csv.")
    parser.add_argument(
        "--assignments-path",
        default=None,
        help="Path to topic assignments CSV (default: outputs/enron/topic_assignments.csv if present, else outputs/topics.csv).",
    )
    parser.add_argument(
        "--topic-words-path",
        default=None,
        help="Path to topic words CSV (default: outputs/enron/topic_words.csv if present, else outputs/topic_words.csv).",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to write word cloud image.")
    parser.add_argument("--topic-col", default=None, help="Topic column override (e.g., topic_id, dominant_topic, topic).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_assignments = (
        project_root / "outputs" / "enron" / "topic_assignments.csv"
        if (project_root / "outputs" / "enron" / "topic_assignments.csv").exists()
        else project_root / "outputs" / "topics.csv"
    )
    default_topic_words = (
        project_root / "outputs" / "enron" / "topic_words.csv"
        if (project_root / "outputs" / "enron" / "topic_words.csv").exists()
        else project_root / "outputs" / "topic_words.csv"
    )

    assignments_path = Path(args.assignments_path) if args.assignments_path else default_assignments
    topic_words_path = Path(args.topic_words_path) if args.topic_words_path else default_topic_words
    out_dir = Path(args.output_dir) if args.output_dir else (
        project_root / "outputs" / ("enron/wordclouds" if "enron" in str(assignments_path) else "wordclouds")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing {assignments_path}.")
    if not topic_words_path.exists():
        raise FileNotFoundError(f"Missing {topic_words_path}.")

    assignments = pd.read_csv(assignments_path)
    topic_words = pd.read_csv(topic_words_path)

    detected_topic_col = args.topic_col
    if detected_topic_col is None:
        for c in ["dominant_topic", "topic_id", "topic"]:
            if c in assignments.columns:
                detected_topic_col = c
                break
    if detected_topic_col is None or detected_topic_col not in assignments.columns:
        raise ValueError("No topic column found in assignments CSV. Use --topic-col.")

    words_topic_col = None
    for c in ["topic_id", "topic", "dominant_topic"]:
        if c in topic_words.columns:
            words_topic_col = c
            break
    if words_topic_col is None:
        raise ValueError("No topic key column found in topic_words CSV.")
    if "top_words" not in topic_words.columns:
        raise ValueError("topic_words CSV must contain a 'top_words' column.")

    topic_counts = assignments[detected_topic_col].value_counts().to_dict()
    topic_words = topic_words.copy()
    topic_words["topic_id"] = topic_words[words_topic_col]

    # add custom stopwords to remove template filler words
    custom_stop = set(STOPWORDS)
    custom_stop |= {
        "dear","sincerely","regards","best","thank","thanks",
        "please","kindly","contact","team","support",
        "account","accounts","information","details","service","services",
        "required","request","requested","update","verification",
        "click","link","access","maintain","maintenance"
    }

    # collect topics first
    topic_rows = topic_words.sort_values("topic_id").to_dict("records")
    n_topics = len(topic_rows)
    if n_topics == 0:
        raise ValueError("No topics available in topic_words CSV.")

    # choose grid size
    ncols = 3
    nrows = math.ceil(n_topics / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax, row in zip(axes, topic_rows):
        topic_id = row["topic_id"]
        weight_multiplier = int(topic_counts.get(topic_id, 1))
        terms = [clean_text(w).replace(" ", "_") for w in str(row["top_words"]).split(",")]
        terms = [t for t in terms if t]
        if not terms:
            ax.axis("off")
            ax.set_title(f"Topic {topic_id}\n(no text)")
            continue

        freqs = {term: (len(terms) - i) * max(weight_multiplier, 1) for i, term in enumerate(terms)}
        wc = WordCloud(
            width=1400,
            height=900,
            background_color="white",
            stopwords=custom_stop,
            collocations=True
        ).generate_from_frequencies(freqs)

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
