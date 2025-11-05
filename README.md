# 🛰️ AI-Powered Misinformation Detector

A Streamlit web app that detects misleading or fake content in text or news URLs.

## 🧠 How it works
- Uses **TF-IDF + Logistic Regression** trained on your labeled dataset.
- Calculates **Reliability = 100 × (1 − P(fake))** (higher = more likely real).

## 📂 Required dataset
Your dataset (`dataset_module1.csv`) should have:
- `text` or `clean_text` → content string
- `label` → 0 = real, 1 = fake

Example:

| clean_text                     | label |
|--------------------------------|-------|
| "the vaccine is safe and works" | 0 |
| "aliens cured covid overnight"  | 1 |

## ▶️ Run locally in VS Code
1. Install requirements  
2. Run:
