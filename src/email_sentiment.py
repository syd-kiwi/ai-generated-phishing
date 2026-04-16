from pathlib import Path
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def main():
    project_root = Path(__file__).resolve().parents[1]
    emails_path = project_root / "outputs" / "emails.parquet"
    out_path = project_root / "outputs" / "sentiment.csv"

    # one time download, safe to call repeatedly
    nltk.download("vader_lexicon")

    sia = SentimentIntensityAnalyzer()

    df = pd.read_parquet(emails_path)
    texts = (df["subject"].fillna("").astype(str) + "\n" + df["raw_text"].fillna("").astype(str))

    scores = texts.apply(lambda t: sia.polarity_scores(t))
    sent_df = pd.DataFrame(list(scores))

    out = pd.concat([df[["email_id", "label"]], sent_df], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print("compound mean:", float(out["compound"].mean()))
    print("compound min:", float(out["compound"].min()))
    print("compound max:", float(out["compound"].max()))
    print(out.head(5))

if __name__ == "__main__":
    main()