import pandas as pd
from langdetect import detect
from tqdm import tqdm
import sqlite3
from pathlib import Path

tqdm.pandas()

def clean_text(text: str) -> str:
    return text.strip()

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

if __name__ == "__main__":
    df = pd.read_csv("data/raw/feedback_sample.csv")
    df["clean_text"] = df["text"].progress_apply(clean_text)
    df["lang"] = df["clean_text"].progress_apply(detect_language)
    df.to_csv("data/processed/feedback_clean.csv", index=False)
    print("✅ Preprocessing complete. Saved to data/processed/feedback_clean.csv")
    print(df.head())

    # Save to SQLite
    DB_PATH = Path("data/ux_feedback.db")
    print("💾 Saving to SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("user_feedback", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Data stored in {DB_PATH} (table: user_feedback)")
