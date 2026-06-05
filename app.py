"""
=============================================================================
app.py — Streamlit App: Sentiment Analysis ChatGPT Play Store Reviews
=============================================================================
Proyek  : Sentiment Analysis Review ChatGPT (Bahasa Indonesia)
Model   : SVM RBF | C=1 | gamma=scale | SBERT paraphrase-multilingual-MiniLM-L12-v2
Test F1-Macro : 60.06% | AUC-Macro : 0.871

CARA JALANKAN:
    streamlit run app.py

STRUKTUR FILE YANG DIBUTUHKAN:
    output/svm_sbert_best_model.joblib
    data/label_classes.npy
    flowchart.png  (opsional)

INSTALL:
    pip install streamlit sentence-transformers scikit-learn joblib numpy pandas
=============================================================================
"""

import re
import os
import time

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# =============================================================================
# PAGE CONFIG — harus paling atas sebelum st lainnya
# =============================================================================

st.set_page_config(
    page_title="Sentiment Analysis - ChatGPT Reviews",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# KONFIGURASI PATH
# =============================================================================

CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(CURRENT_DIR, "src", "output", "svm_sbert_best_model.joblib")
LABEL_CLS_PATH = os.path.join(CURRENT_DIR, "src", "data",   "label_classes.npy")
FLOWCHART_PATH = os.path.join(CURRENT_DIR, "flowchart.png")

SBERT_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
CONF_THRESHOLD = 0.50
MIN_WORDS      = 3

SENTIMENT_COLOR = {
    "positive": "#15803d",
    "neutral":  "#b45309",
    "negative": "#b91c1c",
}

SENTIMENT_BG = {
    "positive": "#f0fdf4",
    "neutral":  "#fffbeb",
    "negative": "#fef2f2",
}

SENTIMENT_BORDER = {
    "positive": "#86efac",
    "neutral":  "#fcd34d",
    "negative": "#fca5a5",
}

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

/* ── base ─────────────────────────────────────────────── */
.stApp {
    background: #f8f7f4;
    color: #1a1a2e;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e4df !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] * {
    color: #4a4a6a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
}

/* ── header ──────────────────────────────────────────── */
.top-header {
    padding: 2.25rem 0 1.75rem 0;
    border-bottom: 1px solid #e8e4df;
    margin-bottom: 2rem;
}
.top-header .title {
    font-family: 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.3px;
    margin: 0;
}
.top-header .subtitle {
    font-size: 0.84rem;
    color: #9090a8;
    margin: 0.4rem 0 0 0;
    font-family: 'JetBrains Mono', monospace;
}
.pill {
    display: inline-block;
    background: #f0eef8;
    border: 1px solid #d8d4f0;
    color: #5a56a0;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 5px;
    margin-top: 8px;
}

/* ── section labels ──────────────────────────────────── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #b0aec8;
    margin-bottom: 10px;
}

/* ── textarea ────────────────────────────────────────── */
.stTextArea textarea {
    background: #ffffff !important;
    color: #1a1a2e !important;
    border: 1px solid #ddd8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    box-shadow: 0 1px 4px rgba(90,86,160,0.06) !important;
}
.stTextArea textarea:focus {
    border-color: #8880d0 !important;
    box-shadow: 0 0 0 3px rgba(136,128,208,0.12) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #c8c4e0 !important;
}

/* ── select ──────────────────────────────────────────── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #ddd8f0 !important;
    border-radius: 10px !important;
    color: #8880c0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── button ──────────────────────────────────────────── */
.stButton > button {
    background: #5a56a0 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
    width: 100%;
    box-shadow: 0 2px 8px rgba(90,86,160,0.25) !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #4a46908 !important;
    box-shadow: 0 4px 14px rgba(90,86,160,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── result card ─────────────────────────────────────── */
.result-block {
    border-radius: 14px;
    padding: 1.6rem 1.9rem;
    margin-top: 0.5rem;
    border: 1.5px solid;
    animation: fadeUp 0.35s ease;
}
.result-label {
    font-family: 'Inter', sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    letter-spacing: -0.5px;
}
.result-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    color: #a0a0b8;
    margin-top: 0.15rem;
}

/* ── probability bars ────────────────────────────────── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
}
.prob-name {
    width: 72px;
    color: #9090a8;
    text-align: right;
}
.prob-bar-bg {
    flex: 1;
    background: #f0eef8;
    border-radius: 4px;
    height: 7px;
    overflow: hidden;
    border: 1px solid #e0dcf0;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
}
.prob-pct {
    width: 44px;
    text-align: right;
    font-weight: 500;
}

/* ── preproc box ─────────────────────────────────────── */
.preproc {
    background: #f4f2fc;
    border: 1px solid #e0dcf4;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
}
.preproc .lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #b0aec8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 5px;
}
.preproc .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.81rem;
    color: #6060a0;
    word-break: break-all;
    line-height: 1.55;
}

/* ── low confidence ──────────────────────────────────── */
.low-conf {
    border: 1px solid #f0c88c;
    background: #fffbf0;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #a07830;
    margin-top: 0.75rem;
    line-height: 1.5;
}

/* ── divider ─────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #e8e4df !important;
    margin: 1.5rem 0 !important;
}

/* ── tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1.5px solid #e8e4df !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #b0aec8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 10px 22px !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: #5a56a0 !important;
    border-bottom-color: #5a56a0 !important;
}

/* ── metric cells ────────────────────────────────────── */
.metric-cell {
    background: #ffffff;
    border: 1px solid #e8e4f0;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(90,86,160,0.05);
}
.metric-cell .v {
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #5a56a0;
}
.metric-cell .l {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.61rem;
    color: #b0aec8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 3px;
}

/* ── info rows ───────────────────────────────────────── */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f0eef8;
    font-size: 0.85rem;
}
.info-row .k {
    color: #9090a8;
    font-size: 0.82rem;
}
.info-row .v {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #5a56a0;
}

/* ── file uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1.5px dashed #d8d4f0 !important;
    border-radius: 12px !important;
}

/* ── dataframe ───────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #e8e4f0 !important;
    border-radius: 10px !important;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(90,86,160,0.05) !important;
}

/* ── download button ─────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    background: #f4f2fc !important;
    color: #5a56a0 !important;
    border: 1px solid #d8d4f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100%;
    transition: all 0.15s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #eae8f8 !important;
    border-color: #b8b4e0 !important;
}

/* ── expander ────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #faf9fe !important;
    border: 1px solid #e8e4f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    color: #5a56a0 !important;
}
.streamlit-expanderContent {
    background: #faf9fe !important;
    border: 1px solid #e8e4f0 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── alert boxes ─────────────────────────────────────── */
.stAlert {
    background: #faf9fe !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PREPROCESSING  — identik dengan preprocessing.py & predict.py
# =============================================================================

def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)

def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", " ", text)

def remove_mojibake(text: str) -> str:
    text = text.replace("\ufffd", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    return text

def normalize_repeated_chars(text: str) -> str:
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_pipeline(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = remove_mojibake(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_repeated_chars(text)
    text = normalize_whitespace(text)
    return text

# =============================================================================
# LOAD — cached agar tidak reload setiap interaksi
# =============================================================================

@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(LABEL_CLS_PATH):
        st.error(f"Label classes tidak ditemukan: {LABEL_CLS_PATH}")
        st.stop()

    svm   = joblib.load(MODEL_PATH)
    lc    = np.load(LABEL_CLS_PATH, allow_pickle=True)
    sbert = SentenceTransformer(SBERT_MODEL)
    return svm, sbert, lc

# =============================================================================
# PREDICT HELPERS
# =============================================================================

def predict_single(text: str, model, sbert, label_classes):
    clean = clean_pipeline(text)
    if not clean:
        return None

    emb = sbert.encode(
        [clean], batch_size=32,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    idx   = model.predict(emb)[0]
    label = str(label_classes[idx])
    proba = {}

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(emb)[0]
        proba = {str(label_classes[i]): float(p[i]) for i in range(len(label_classes))}

    conf = max(proba.values()) if proba else None
    return {"label": label, "confidence": conf, "proba": proba,
            "clean": clean, "word_count": len(clean.split())}


def predict_batch(texts: list, model, sbert, label_classes):
    cleaned   = [clean_pipeline(t) for t in texts]
    valid_idx = [i for i, c in enumerate(cleaned) if c]

    if not valid_idx:
        return [], []

    embs = sbert.encode(
        [cleaned[i] for i in valid_idx],
        batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    preds  = model.predict(embs)
    labels = [str(label_classes[p]) for p in preds]
    pmatrix = model.predict_proba(embs) if hasattr(model, "predict_proba") else None

    rows = []
    for i, vi in enumerate(valid_idx):
        pd_ = ({str(label_classes[j]): float(pmatrix[i][j])
                for j in range(len(label_classes))} if pmatrix is not None else {})
        conf = max(pd_.values()) if pd_ else None
        rows.append({
            "review_original": texts[vi],
            "review_clean":    cleaned[vi],
            "word_count":      len(cleaned[vi].split()),
            "prediction":      labels[i],
            "confidence":      round(conf, 4) if conf else None,
            "conf_negative":   round(pd_.get("negative", 0), 4),
            "conf_neutral":    round(pd_.get("neutral",  0), 4),
            "conf_positive":   round(pd_.get("positive", 0), 4),
            "low_confidence":  (conf < CONF_THRESHOLD) if conf else True,
        })
    return rows, valid_idx

# =============================================================================
# RENDER HELPERS
# =============================================================================

def render_prob_bar(label: str, pct: float, color: str):
    w = f"{pct*100:.1f}%"
    st.markdown(f"""
<div class="prob-row">
    <span class="prob-name">{label}</span>
    <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{w}; background:{color}"></div>
    </div>
    <span class="prob-pct" style="color:{color}">{w}</span>
</div>""", unsafe_allow_html=True)

# =============================================================================
# LOAD ASSETS
# =============================================================================

model, sbert, label_classes = load_assets()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("**Sentiment Analyzer**")
    st.markdown("---")
    st.markdown("""
```
Model    : SVM RBF
C        : 1
gamma    : scale
SBERT    : 384-dim
F1-Macro : 60.06%
AUC      : 0.871
```
""")
    st.markdown("---")
    st.markdown("**Kelas**")
    st.markdown("""
```
0  negative
1  neutral
2  positive
```
""")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("""
```
scrape.py
preprocessing.py
spliting.py
embedding.py
modeling&tunning.py
evaluasi.py
predict.py
app.py  <-- ini
```
""")
    st.markdown("---")
    st.caption("NoLimit Indonesia · DS Test")

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="top-header">
    <p class="title">Sentiment Analysis - ChatGPT Play Store Reviews</p>
    <p class="subtitle">Bahasa Indonesia · SVM + SBERT · 3-class</p>
    <div>
        <span class="pill">SVM RBF</span>
        <span class="pill">SBERT 384-dim</span>
        <span class="pill">F1-Macro 60.06%</span>
        <span class="pill">AUC 0.871</span>
        <span class="pill">seed=42</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="sec-label">Review Input</div>', unsafe_allow_html=True)

        examples = [
            "-- pilih contoh --",
            "Aplikasi ini bagus banget, sangat membantu pekerjaan saya sehari-hari!",
            "Biasa aja sih, fiturnya standar tidak ada yang istimewa.",
            "Baterai hape jadi cepet habis gara-gara app ini, sangat mengecewakan.",
            "BAGUSSS BANGET!!! Suka sekali sama fitur barunya",
            "Lumayan lah buat chatting santai tapi kalau buat kerja kurang optimal.",
            "Aplikasi sering crash dan loading lama banget, bikin frustrasi.",
        ]
        chosen = st.selectbox(
            "Contoh review",
            examples,
            label_visibility="collapsed",
            key="example_select"
        )

        default = "" if chosen == "-- pilih contoh --" else chosen
        review_input = st.text_area(
            "Review",
            value=default,
            height=150,
            placeholder="Ketik review Bahasa Indonesia...",
            label_visibility="collapsed",
            key="review_text"
        )

        predict_btn = st.button("Predict", use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-label">Hasil</div>', unsafe_allow_html=True)

        if predict_btn:
            if not review_input.strip():
                st.warning("Review tidak boleh kosong.")
            else:
                with st.spinner("Analyzing..."):
                    t0  = time.time()
                    res = predict_single(review_input.strip(), model, sbert, label_classes)
                    dur = time.time() - t0

                if res is None:
                    st.error("Review tidak valid setelah preprocessing.")
                else:
                    lbl    = res["label"]
                    color  = SENTIMENT_COLOR.get(lbl, "#888")
                    bg     = SENTIMENT_BG.get(lbl, "#111")
                    border = SENTIMENT_BORDER.get(lbl, "#333")
                    conf   = res["confidence"]

                    # Result block
                    st.markdown(f"""
<div class="result-block" style="background:{bg}; border-color:{border}">
    <div class="result-label" style="color:{color}">{lbl.upper()}</div>
    <div class="result-meta">
        confidence: {conf:.1%} &nbsp;&nbsp; words: {res['word_count']} &nbsp;&nbsp; {dur*1000:.0f}ms
    </div>
</div>
""", unsafe_allow_html=True)

                    # Probability bars
                    if res["proba"]:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="sec-label">Probabilitas</div>', unsafe_allow_html=True)
                        for lbl_name in ["positive", "neutral", "negative"]:
                            if lbl_name in res["proba"]:
                                render_prob_bar(
                                    lbl_name,
                                    res["proba"][lbl_name],
                                    SENTIMENT_COLOR.get(lbl_name, "#888"),
                                )

                    # Low confidence notice
                    if conf and conf < CONF_THRESHOLD:
                        st.markdown(f"""
<div class="low-conf">
    low confidence: {conf:.1%} &lt; {CONF_THRESHOLD:.0%}<br>
    Review mungkin ambigu, sarkasme, atau campuran bahasa.
</div>
""", unsafe_allow_html=True)

                    # Preprocessing transparency
                    st.markdown(f"""
<div class="preproc">
    <div class="lbl">after preprocessing</div>
    <div class="val">{res['clean']}</div>
</div>
""", unsafe_allow_html=True)

        else:
            st.markdown("""
<div style="padding:3rem 0; color:#c8c4e0; font-family:'JetBrains Mono',monospace; font-size:0.8rem; text-align:center; line-height:2">
    -- no prediction yet --<br>
    ketik review lalu klik Predict
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Format CSV</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#9090a8;
     background:#f4f2fc; border:1px solid #e0dcf4; border-radius:10px;
     padding:0.9rem 1.1rem; margin-bottom:1rem; line-height:1.8">
    Kolom wajib: <span style="color:#5a56a0">review</span><br>
    Kolom lain boleh ada (rating, review_id, ...) dan akan diabaikan.<br>
    <br>
    Contoh isi CSV:<br>
    <span style="color:#5a56a0; font-weight:500">review</span><br>
    <span style="color:#7070a0">Aplikasi bagus banget</span><br>
    <span style="color:#7070a0">Sering error tidak rekomen</span><br>
    <span style="color:#7070a0">Biasa aja standar</span>
</div>
""", unsafe_allow_html=True)

    # Template download
    template_df = pd.DataFrame({
        "review": [
            "Aplikasi ini sangat membantu pekerjaan saya sehari-hari",
            "Biasa aja, tidak ada yang spesial menurut saya",
            "Sering error dan baterai cepat habis, tidak rekomen",
            "BAGUSSS fitur barunya keren banget",
            "Lumayan lah buat santai tapi buat kerja kurang",
        ]
    })
    st.download_button(
        label="Download template CSV",
        data=template_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="template_reviews.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Upload CSV</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception:
            try:
                uploaded.seek(0)
                df_up = pd.read_csv(uploaded, encoding="utf-8")
            except Exception as e:
                st.error(f"Gagal baca CSV: {e}")
                df_up = None

        if df_up is not None:
            if "review" not in df_up.columns:
                st.error(f"Kolom 'review' tidak ditemukan. Tersedia: {list(df_up.columns)}")
            else:
                st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#9090a8; margin-bottom:1rem">
    {len(df_up)} baris ditemukan
</div>
""", unsafe_allow_html=True)

                if st.button("Predict CSV", use_container_width=True):
                    with st.spinner(f"Memproses {len(df_up)} review..."):
                        texts = df_up["review"].fillna("").tolist()
                        rows, valid_idx = predict_batch(texts, model, sbert, label_classes)

                    if not rows:
                        st.warning("Tidak ada review valid untuk diprediksi.")
                    else:
                        df_res = pd.DataFrame(rows)
                        df_res.insert(0, "id", range(1, len(df_res) + 1))

                        # Ringkasan
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="sec-label">Ringkasan</div>', unsafe_allow_html=True)

                        dist   = df_res["prediction"].value_counts()
                        n_low  = int(df_res["low_confidence"].sum())
                        avg_cf = df_res["confidence"].mean()

                        c1, c2, c3, c4, c5 = st.columns(5)
                        for col_obj, lbl_name in zip([c1, c2, c3], ["positive", "neutral", "negative"]):
                            cnt = int(dist.get(lbl_name, 0))
                            pct = cnt / len(df_res) * 100
                            clr = SENTIMENT_COLOR.get(lbl_name, "#888")
                            with col_obj:
                                st.markdown(f"""
<div class="metric-cell">
    <div class="v" style="color:{clr}">{cnt}</div>
    <div class="l">{lbl_name} ({pct:.0f}%)</div>
</div>
""", unsafe_allow_html=True)

                        with c4:
                            st.markdown(f"""
<div class="metric-cell">
    <div class="v">{avg_cf:.1%}</div>
    <div class="l">avg conf</div>
</div>
""", unsafe_allow_html=True)
                        with c5:
                            st.markdown(f"""
<div class="metric-cell">
    <div class="v" style="color:#806030">{n_low}</div>
    <div class="l">low conf</div>
</div>
""", unsafe_allow_html=True)

                        # Tabel
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="sec-label">Hasil Prediksi</div>', unsafe_allow_html=True)
                        st.dataframe(
                            df_res[["id","review_original","review_clean","prediction",
                                    "confidence","conf_negative","conf_neutral","conf_positive","low_confidence"]],
                            use_container_width=True,
                            height=340,
                        )

                        # Download
                        st.download_button(
                            label="Download Results",
                            data=df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="sec-label">Performa</div>', unsafe_allow_html=True)
        for name, val, clr in [
            ("Test F1-Macro",   "60.06%", "#8080c0"),
            ("Test Accuracy",   "~80%",   "#60a060"),
            ("AUC-Macro (OvR)", "0.871",  "#a0a040"),
            ("F1 Positive",     "~88%",   "#16a34a"),
            ("F1 Negative",     "~65%",   "#dc2626"),
            ("F1 Neutral",      "26.74%", "#d97706"),
        ]:
            st.markdown(f"""
<div class="info-row">
    <span class="k">{name}</span>
    <span class="v" style="color:{clr}">{val}</span>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-label">Konfigurasi</div>', unsafe_allow_html=True)
        for k, v in [
            ("Algoritma",    "SVC"),
            ("Kernel",       "RBF"),
            ("C",            "1"),
            ("gamma",        "scale"),
            ("class_weight", "balanced"),
            ("probability",  "True (Platt Scaling)"),
            ("SBERT model",  "paraphrase-multilingual-MiniLM-L12-v2"),
            ("Dimensi",      "384-dim L2-normalized"),
            ("Tuning",       "GridSearchCV StratifiedKFold-5"),
            ("Seed",         "42"),
        ]:
            st.markdown(f"""
<div class="info-row">
    <span class="k">{k}</span>
    <span class="v">{v}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Keputusan Teknis</div>', unsafe_allow_html=True)

    for title, body in [
        ("Mengapa SBERT?",
         "SBERT paraphrase-multilingual mendukung Bahasa Indonesia secara native. "
         "Output 384-dim L2-normalized sangat efisien dengan SVM karena inner product di ruang unit sphere. "
         "Model ini jauh lebih ringan dari BERT 768-dim dengan performa kompetitif untuk teks pendek."),
        ("Mengapa tidak Stopword Removal / Stemming?",
         "SBERT memproses konteks penuh — stopword seperti 'tidak', 'tapi', 'kurang' justru penting "
         "untuk embedding semantik. Stemming merusak representasi subword yang dipelajari model."),
        ("Mengapa SMOTE tidak dipakai?",
         "Eksperimen menunjukkan SMOTE menurunkan F1-Macro dari 60.06% ke 53.70%. "
         "Kelas neutral rendah bukan karena kurang data, tapi karena ambiguitas label "
         "(rating=3 mencakup review beragam dari 'biasa' hingga 'cukup puas')."),
        ("Mengapa RBF menang dari Linear?",
         "RBF C=1 gamma=scale lebih baik karena sentimen Bahasa Indonesia mengandung pola non-linear: "
         "sarkasme ('bagus banget padahal error terus'), campuran kata gaul, singkatan informal."),
    ]:
        with st.expander(title):
            st.markdown(f"""
<div style="font-family:'Inter',sans-serif; font-size:0.85rem;
     color:#5a5880; line-height:1.7; padding:0.5rem 0">
    {body}
</div>
""", unsafe_allow_html=True)

    # Flowchart
    if os.path.exists(FLOWCHART_PATH):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Pipeline Flowchart</div>', unsafe_allow_html=True)
        st.image(FLOWCHART_PATH, use_container_width=True)
    else:
        st.markdown("""
<div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#b0aec8;
     padding:1rem; border:1.5px dashed #d8d4f0; border-radius:10px; text-align:center; background:#faf9fe">
    flowchart.png not found — tambahkan ke root folder
</div>
""", unsafe_allow_html=True)