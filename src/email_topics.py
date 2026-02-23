from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def main():
    project_root = Path(__file__).resolve().parents[1]
    emails_path = project_root / "outputs" / "emails.parquet"
    topics_out = project_root / "outputs" / "topics.csv"
    words_out = project_root / "outputs" / "topic_words.csv"

    df = pd.read_parquet(emails_path)
    texts = (df["subject"].fillna("").astype(str) + "\n" + df["raw_text"].fillna("").astype(str))

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=3,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)

    n_topics = 8
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    doc_topic = lda.fit_transform(X)

    topic_id = doc_topic.argmax(axis=1)
    df_topics = pd.DataFrame({
        "email_id": df["email_id"],
        "label": df["label"],
        "topic_id": topic_id
    })

    words = vectorizer.get_feature_names_out()
    top_words_rows = []
    for k in range(n_topics):
        comps = lda.components_[k]
        top_idx = comps.argsort()[-12:][::-1]
        top_words = [words[i] for i in top_idx]
        top_words_rows.append({"topic_id": k, "top_words": ", ".join(top_words)})

    topic_words = pd.DataFrame(top_words_rows)

    topics_out.parent.mkdir(parents=True, exist_ok=True)
    df_topics.to_csv(topics_out, index=False)
    topic_words.to_csv(words_out, index=False)

    print(f"Wrote {topics_out}")
    print(f"Wrote {words_out}")
    print(topic_words)

if __name__ == "__main__":
    main()