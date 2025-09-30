import pandas as pd
from langdetect import detect
from tqdm import tqdm

tqdm.pandas()  # enable progress bars

def clean_text(text: str) -> str:
    """Basic text cleaning (strip whitespace)."""
    return text.strip()

def detect_language(text: str) -> str:
    """Detect language of the text."""
    try:
        return detect(text)
    except:
        return "unknown"

if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv("data/raw/feedback_sample.csv")

    # Clean + detect language
    df["clean_text"] = df["text"].progress_apply(clean_text)
    df["lang"] = df["clean_text"].progress_apply(detect_language)

    # Save processed data
    df.to_csv("data/processed/feedback_clean.csv", index=False)

    print("✅ Preprocessing complete. Saved to data/processed/feedback_clean.csv")
    print(df.head())
