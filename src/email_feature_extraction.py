from pathlib import Path
import re
import pandas as pd

URL_RE = re.compile(r"(?i)\b(?:hxxps?|https?)://[^\s<>'\"]+|\bwww\.[^\s<>'\"]+")
OBFUSCATION_RE = re.compile(r"(?i)\bhxxp\b|\s+dot\s+|\[.\]|\(.\)")

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())

def extract_urls(raw: str) -> list[str]:
    urls = []
    for u in URL_RE.findall(raw):
        u = u.strip(").,;!]?\"'")
        if u.lower().startswith("www."):
            u = "http://" + u
        u = re.sub(r"(?i)^hxxp", "http", u)
        urls.append(u)
    return urls

def count_terms(tokens: list[str], terms: list[str]) -> int:
    return sum(tokens.count(t) for t in terms)

def phrase_hits(text_norm: str, phrases: list[str]) -> int:
    joined = " " + text_norm + " "
    return sum(joined.count(" " + p + " ") for p in phrases)

def main():
    project_root = Path(__file__).resolve().parents[1]
    emails_path = project_root / "outputs" / "emails.parquet"
    out_path = project_root / "outputs" / "features.csv"

    df = pd.read_parquet(emails_path)

    triggers = {
        "urgency": {
            "tokens": ["urgent","immediate","immediately","now","today","asap","prompt","promptly","soon","deadline","final","notice","action","critical","crucial"],
            "phrases": ["immediate action", "action needed", "action requested", "prompt attention", "as soon as possible", "do not delay"]
        },
        "fear_loss": {
            "tokens": ["suspended","locked","compromised","unauthorized","breach","risk","risks","penalty","security","disruptions","interruptions","issues","threat","threats"],
            "phrases": ["prevent unauthorized access", "avoid any potential", "prevent any potential", "without interruptions", "security risk", "security risks"]
        },
        "authority": {
            "tokens": ["team","support","helpdesk","administrator","official","compliance","policy","management"],
            "phrases": ["support team", "account management", "account holder", "customer support"]
        },
        "action_demand": {
            "tokens": ["verify","confirm","update","click","access","login","sign","signin","reset","validate","complete","proceed","required","ensure","maintain"],
            "phrases": ["click on the following link", "access the following link", "update your account", "verify your account", "complete the required"]
        },
        "account_focus": {
            "tokens": ["account","password","credentials","identity","information","details","services","functionality","integrity","accuracy"],
            "phrases": ["account information", "account details", "account update", "account verification"]
        },
        "reward_gain": {
            "tokens": ["winner","prize","reward","gift","bonus","free","selected","claim"],
            "phrases": ["you have been selected", "claim your reward", "limited time offer"]
        }
    }

    out_rows = []
    for _, r in df.iterrows():
        raw = str(r.get("raw_text", ""))
        subject = str(r.get("subject", ""))

        combined = subject + "\n" + raw
        norm = normalize(combined)
        toks = tokenize(norm)
        urls = extract_urls(combined)

        feats = {
            "email_id": r["email_id"],
            "label": r["label"],
            "word_count": len(toks),
            "char_count": len(raw),
            "subject_len": len(subject),
            "url_count": len(urls),
            "has_url": int(len(urls) > 0),
            "has_obfuscation": int(bool(OBFUSCATION_RE.search(combined))),
        }

        for name, rule in triggers.items():
            tok_hits = count_terms(toks, rule["tokens"])
            phr_hits = phrase_hits(norm, rule["phrases"])
            score = tok_hits + 2 * phr_hits

            feats[f"{name}_token_hits"] = tok_hits
            feats[f"{name}_phrase_hits"] = phr_hits
            feats[f"{name}_score"] = score
            feats[f"{name}_rate_per_100w"] = 100.0 * score / max(feats["word_count"], 1)

        out_rows.append(feats)

    feats_df = pd.DataFrame(out_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print("n_rows:", len(feats_df))
    print("pct_has_url:", float(feats_df["has_url"].mean()))
    print("mean_url_count:", float(feats_df["url_count"].mean()))
    print("mean_urgency_rate_per_100w:", float(feats_df["urgency_rate_per_100w"].mean()))
    print("mean_action_rate_per_100w:", float(feats_df["action_demand_rate_per_100w"].mean()))

if __name__ == "__main__":
    main()