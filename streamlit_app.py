# streamlit_app.py
# Streamlit UI for AI-Powered Misinformation Detection

import os
import re
import html
import requests
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

# ---------- CONFIG ----------
CSV_PATH = "dataset_module1.csv"       # CSV must have 'text' or 'clean_text' + 'label'
MODEL_PATH = "model.joblib"
MAX_FEATURES = 50_000


# ---------- TEXT CLEANING ----------
def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def fetch_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (misinfo-detector/1.0)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return " ".join(ps)


def reliability_score(p_fake: float) -> float:
    return float(np.clip(100.0 * (1.0 - p_fake), 0.0, 100.0))


def pick_text_column(df: pd.DataFrame) -> pd.Series:
    if "text" in df.columns:
        return df["text"]
    if "clean_text" in df.columns:
        return df["clean_text"]
    raise ValueError("CSV must have a 'text' or 'clean_text' column.")


# ---------- MODEL ----------
@st.cache_resource(show_spinner=True)
def load_or_train_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("dataset_module1.csv not found. Upload it first.")

    df = pd.read_csv(CSV_PATH).dropna()
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column (0 = real, 1 = fake).")

    X = pick_text_column(df).astype(str).apply(clean_text)
    y = df["label"].astype(int)

    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipe.fit(x_tr, y_tr)
    proba = pipe.predict_proba(x_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    st.session_state["_last_auc"] = float(auc)

    joblib.dump(pipe, MODEL_PATH)
    return pipe


def score_texts(model: Pipeline, texts: list) -> pd.DataFrame:
    cleaned = [clean_text(t) for t in texts]
    p_fake = model.predict_proba(cleaned)[:, 1]
    rel = [reliability_score(p) for p in p_fake]
    label = ["Fake" if p >= 0.5 else "Real" for p in p_fake]
    return pd.DataFrame({
        "input": texts,
        "p_fake": p_fake,
        "reliability": rel,
        "prediction": label
    })


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Misinformation Detector", page_icon="🛰️", layout="centered")

st.title("🛰️ Misinformation Detector")
st.caption("A machine learning approach to spotting misleading information online.")

with st.expander("Model Info"):
    st.write("- TF-IDF + Logistic Regression baseline.")
    st.write("- Retrains automatically if no model file found.")
    if "_last_auc" in st.session_state:
        st.metric("Validation AUC", f"{st.session_state['_last_auc']:.3f}")

tab1, tab2 = st.tabs(["🔤 Analyze Text", "🌐 Analyze URL"])

with tab1:
    user_text = st.text_area("Enter text to analyze:", height=200)
    if st.button("Analyze Text"):
        if user_text.strip():
            model = load_or_train_model()
            df = score_texts(model, [user_text.strip()])
            row = df.iloc[0]
            st.metric("Reliability", f"{row['reliability']:.1f}%")
            st.write(f"Prediction: **{row['prediction']}** (P(fake)={row['p_fake']:.2f})")
        else:
            st.warning("Please enter some text.")

with tab2:
    url = st.text_input("Enter a news/article URL:")
    if st.button("Analyze URL"):
        if url.strip():
            try:
                raw = fetch_url(url.strip())
                model = load_or_train_model()
                df = score_texts(model, [raw])
                row = df.iloc[0]
                st.metric("Reliability", f"{row['reliability']:.1f}%")
                st.write(f"Prediction: **{row['prediction']}** (P(fake)={row['p_fake']:.2f})")
                with st.expander("Show extracted text"):
                    st.write(raw[:3000] + ("..." if len(raw) > 3000 else ""))
            except Exception as e:
                st.error(f"Could not analyze URL: {e}")
        else:
            st.warning("Please enter a valid URL.")

st.markdown("---")
st.caption("Reliability = 100 × (1 − P(fake)). Higher = more likely real.")
