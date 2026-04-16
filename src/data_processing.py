from pathlib import Path
import re
import pandas as pd

HEADER_RE = re.compile(r"(?im)^\s*header\s*:\s*(.+)$")

def read_email_file(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()

    m = HEADER_RE.search(raw)
    subject = m.group(1).strip() if m else ""

    return {
        "email_id": path.stem,
        "subject": subject,
        "raw_text": raw,
        "label": "ai_phish",
        "source_file": str(path.name),
    }

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"

    files = sorted([p for p in data_root.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    print("project_root:", project_root)
    print("data_root:", data_root)
    print("txt files found:", len(files))

    if not files:
        raise RuntimeError(f"No .txt files found in {data_root}")

    rows = [read_email_file(p) for p in files]
    df = pd.DataFrame(rows)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    df.to_parquet("outputs/emails.parquet", index=False)

    print("Wrote outputs/emails.parquet")
    print("n_emails:", len(df))
    print("header_found_pct:", (df["subject"].str.len() > 0).mean())
    print("example_header:", df["subject"].iloc[0])

if __name__ == "__main__":
    main()