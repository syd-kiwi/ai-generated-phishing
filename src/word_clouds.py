from pathlib import Path
import re
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
    project_root = Path(__file__).resolve().parents[1]
    emails_path = project_root / "outputs" / "emails.parquet"
    topics_path = project_root / "outputs" / "topics.csv"
    out_dir = project_root / "outputs" / "wordclouds"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emails_path.exists():
        raise FileNotFoundError(f"Missing {emails_path}")
    if not topics_path.exists():
        raise FileNotFoundError(f"Missing {topics_path}. Run src/03_topics.py first.")

    emails = pd.read_parquet(emails_path)
    topics = pd.read_csv(topics_path)

    df = emails.merge(topics, on=["email_id", "label"], how="inner")

    # combine subject and body so urgency phrases in headers are included
    df["text"] = df["subject"].fillna("").astype(str) + " " + df["raw_text"].fillna("").astype(str)
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