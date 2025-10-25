import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Sentiment (multilingual) – small & CPU-friendly
from transformers import pipeline

# Lightweight topic modeling (no GPU)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# ---------------------- config ----------------------
DB_PATH = Path("data/ux_feedback.db")
IN_TABLE = "user_feedback"
OUT_TABLE = "user_feedback_enriched"

ENRICHED_DIR = Path("data/enriched")
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_CSV = ENRICHED_DIR / "enriched_feedback.csv"

TEXT_COL_CANDIDATES = ["translated_en", "clean_text", "text"]  # choose best available

MAX_DOCS_FOR_DEMO = None  # set to e.g. 200 if you want to cap during testing
BATCH_SIZE = 16  # sentiment batching

# Topic modeling settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 1  # ignore rare terms
# N topics scales with data size
def choose_n_topics(n_docs: int) -> int:
    if n_docs < 50:
        return 3
    elif n_docs < 200:
        return 6
    else:
        return 8
# ----------------------------------------------------


def pick_text_column(df: pd.DataFrame) -> str:
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"None of {TEXT_COL_CANDIDATES} found. Available: {list(df.columns)}")


def build_sentiment_pipeline():
    # Multilingual star ratings (1–5). Small and works on CPU.
    # Model card: nlptown/bert-base-multilingual-uncased-sentiment
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=-1,  # CPU
        truncation=True
    )


def map_star_to_label_and_score(label: str) -> Tuple[str, int]:
    # Map "1 star" .. "5 stars" -> label + score
    try:
        stars = int(label.split()[0])
    except Exception:
        stars = 3
    if stars <= 2:
        return ("negative", stars - 3)  # -2 or -1
    elif stars == 3:
        return ("neutral", 0)
    else:
        return ("positive", stars - 3)  # +1 or +2


def run_sentiment(texts: List[str]) -> Tuple[List[str], List[int], List[float]]:
    clf = build_sentiment_pipeline()
    labels, scores, confidences = [], [], []
    # batch predict
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Sentiment"):
        batch = texts[i:i+BATCH_SIZE]
        preds = clf(batch)
        for p in preds:
            label_txt, score_int = map_star_to_label_and_score(p["label"])
            labels.append(label_txt)
            scores.append(score_int)
            confidences.append(float(p.get("score", 0.0)))
    return labels, scores, confidences


def run_topic_modeling(texts: List[str], n_topics: int):
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)  # doc-topic weights
    H = nmf.components_       # topic-term weights
    vocab = np.array(vectorizer.get_feature_names_out())

    # assign dominant topic per doc
    doc_topics = W.argmax(axis=1)

    # top keywords per topic
    topic_keywords = []
    TOP_K = 8
    for k in range(n_topics):
        top_idx = H[k].argsort()[::-1][:TOP_K]
        words = vocab[top_idx]
        topic_keywords.append(", ".join(words))

    return doc_topics, topic_keywords


def main():
    print("🔌 Loading data from SQLite...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {IN_TABLE}", conn)

    if MAX_DOCS_FOR_DEMO:
        df = df.head(MAX_DOCS_FOR_DEMO)

    text_col = pick_text_column(df)
    print(f"📝 Using text column: '{text_col}'")

    texts = df[text_col].fillna("").astype(str).tolist()

    # --- Sentiment ---
    print("🧠 Running sentiment (multilingual, CPU)...")
    sent_labels, sent_scores, sent_conf = run_sentiment(texts)
    df["sentiment_label"] = sent_labels
    df["sentiment_score"] = sent_scores
    df["sentiment_confidence"] = sent_conf

    # --- Topic modeling ---
    print("🧩 Running topic modeling (TF-IDF + NMF)...")
    # use English stopwords even for mixed text; adequate for quick clustering
    n_topics = choose_n_topics(len(df))
    doc_topics, topic_keywords = run_topic_modeling(texts, n_topics)
    df["topic_id"] = doc_topics
    # map topic_id -> keywords string
    topic_kw_map = {i: kw for i, kw in enumerate(topic_keywords)}
    df["topic_keywords"] = df["topic_id"].map(topic_kw_map)

    # --- Save outputs ---
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENRICHED_CSV, index=False)
    print(f"💾 Saved enriched CSV -> {ENRICHED_CSV}")

    df.to_sql(OUT_TABLE, conn, if_exists="replace", index=False)
    conn.close()
    print(f"💾 Saved to SQLite -> {DB_PATH} (table: {OUT_TABLE})")

    # --- quick summary print ---
    print("\n📊 Sentiment distribution:")
    print(df["sentiment_label"].value_counts())

    print("\n🗂 Topics (id : keywords):")
    for tid, kw in topic_kw_map.items():
        print(f"  {tid}: {kw}")

    print("\n✅ ML enrichment complete.")


if __name__ == "__main__":
    main()
