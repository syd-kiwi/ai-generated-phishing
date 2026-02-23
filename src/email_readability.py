from pathlib import Path
import pandas as pd
import re
import textstat

def count_sentences(text: str) -> int:
    # basic sentence split
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return max(len(sentences), 1)

def count_words(text: str) -> int:
    words = re.findall(r"[a-zA-Z0-9']+", text)
    return max(len(words), 1)

def lexical_diversity(text: str) -> float:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def main():
    project_root = Path(__file__).resolve().parents[1]
    emails_path = project_root / "outputs" / "emails.parquet"
    out_path = project_root / "outputs" / "readability.csv"

    df = pd.read_parquet(emails_path)

    # Combine subject + body
    df["combined"] = df["subject"].fillna("").astype(str) + "\n" + df["raw_text"].fillna("").astype(str)

    results = []

    for _, row in df.iterrows():
        text = str(row["combined"])

        words = count_words(text)
        sentences = count_sentences(text)

        avg_sentence_len = words / sentences

        grade_level = textstat.flesch_kincaid_grade(text)

        lex_div = lexical_diversity(text)

        results.append({
            "email_id": row["email_id"],
            "label": row["label"],
            "word_count": words,
            "sentence_count": sentences,
            "avg_sentence_length": avg_sentence_len,
            "flesch_kincaid_grade": grade_level,
            "lexical_diversity": lex_div
        })

    out_df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print("Mean Grade Level:", out_df["flesch_kincaid_grade"].mean())
    print("Mean Avg Sentence Length:", out_df["avg_sentence_length"].mean())
    print("Mean Lexical Diversity:", out_df["lexical_diversity"].mean())
    print(out_df.head())

if __name__ == "__main__":
    main()