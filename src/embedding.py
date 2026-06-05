"""
FEATURE EXTRACTION — SBERT Embedding
======================================
INPUT  :
    data/train.csv
    data/test.csv

OUTPUT :
    data/X_train_emb.npy   — Embedding train (shape: N_train x 384)
    data/X_test_emb.npy    — Embedding test  (shape: N_test  x 384)
    data/y_train.npy       — Label train (encoded integer)
    data/y_test.npy        — Label test  (encoded integer)
    data/label_classes.npy — Mapping index -> nama kelas

MODEL  : paraphrase-multilingual-MiniLM-L12-v2 (384 dim)
  - Support Bahasa Indonesia
  - Ringan & cepat (vs 768-dim models)
  - Digunakan luas di riset NLP Indonesia
  - Dimensi 384 sudah cukup untuk dataset ~6000 review

NORMALIZATION:
  - normalize_embeddings=True -> L2 normalization ke unit vector
  - Berdasarkan riset: L2-norm + LinearSVM adalah pipeline
    paling stabil untuk SBERT embeddings (Amponsah-Kaakyire et al.)
  - Normalize di sisi encode() agar konsisten antara train & test
    (bukan normalize setelah split terpisah)

REPRODUCIBILITY:
  - SEED = 42 diset untuk random, numpy, dan torch
  - Memastikan hasil konsisten antar run

DTYPE:
  - Embedding di-cast ke float32 setelah encode
  - SBERT sudah output float32 by default, cast ini bersifat safeguard
  - Hemat memori disk (~50% vs float64) saat menyimpan .npy
  - Catatan: sklearn SVC akan auto-cast ke float64 saat .fit(),
    hal ini normal dan tidak mempengaruhi akurasi model

INSTALL:
  pip install sentence-transformers pandas numpy scikit-learn
"""

import os
import sys
import random                  # [TAMBAHAN] untuk random seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ================================================================
# REPRODUCIBILITY — SEED                                [TAMBAHAN]
# ================================================================
# Set seed di awal sebelum import torch & model load
# Memastikan hasil embedding konsisten antar run
# Referensi: PyTorch reproducibility best practice

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    print(f"  [SEED] random={SEED}, numpy={SEED}, torch={SEED} — reproducibility aktif")
except ImportError:
    print(f"  [SEED] random={SEED}, numpy={SEED} — torch tidak ditemukan, skip torch seed")

# ================================================================
# KONFIGURASI
# ================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(CURRENT_DIR, "data")

TRAIN_FILE  = os.path.join(DATA_DIR, "train.csv")
TEST_FILE   = os.path.join(DATA_DIR, "test.csv")

# Output embedding files
X_TRAIN_EMB = os.path.join(DATA_DIR, "X_train_emb.npy")
X_TEST_EMB  = os.path.join(DATA_DIR, "X_test_emb.npy")
Y_TRAIN     = os.path.join(DATA_DIR, "y_train.npy")
Y_TEST      = os.path.join(DATA_DIR, "y_test.npy")
LABEL_CLS   = os.path.join(DATA_DIR, "label_classes.npy")

MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COL    = "review_text"
LABEL_COL   = "sentiment_label"
BATCH_SIZE  = 32    # Aman untuk CPU maupun GPU dengan RAM 8GB+

# ================================================================
# HELPERS
# ================================================================

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def safe_print(text: str, max_len: int = 100):
    enc  = sys.stdout.encoding or "utf-8"
    safe = str(text).encode(enc, errors="replace").decode(enc)
    print(f"  {safe[:max_len]}{'...' if len(safe) > max_len else ''}")


# ================================================================
# MAIN
# ================================================================

def main():

    # ----------------------------------------------------------
    # STEP 1 - Load train & test CSV
    # ----------------------------------------------------------
    print_section("LOAD DATASET")

    for path in [TRAIN_FILE, TEST_FILE]:
        if not os.path.exists(path):
            print(f"  ERROR: File tidak ditemukan: {path}")
            print(f"  Pastikan splitting.py sudah dijalankan terlebih dahulu.")
            return

    df_train = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
    df_test  = pd.read_csv(TEST_FILE,  encoding="utf-8-sig")

    print(f"  train.csv : {len(df_train)} review")
    print(f"  test.csv  : {len(df_test)} review")

    X_train_text = df_train[TEXT_COL].fillna("").tolist()
    X_test_text  = df_test[TEXT_COL].fillna("").tolist()
    y_train_raw  = df_train[LABEL_COL].tolist()
    y_test_raw   = df_test[LABEL_COL].tolist()

    # ----------------------------------------------------------
    # STEP 2 - Label Encoding
    # negative -> 0 | neutral -> 1 | positive -> 2
    # ----------------------------------------------------------
    print_section("LABEL ENCODING")

    le = LabelEncoder()
    le.fit(y_train_raw)       # fit hanya pada train
    y_train_enc = le.transform(y_train_raw)
    y_test_enc  = le.transform(y_test_raw)

    print(f"  Mapping label:")
    for idx, cls in enumerate(le.classes_):
        count_tr = (y_train_enc == idx).sum()
        count_te = (y_test_enc  == idx).sum()
        enc = sys.stdout.encoding or "utf-8"
        cls_s = str(cls).encode(enc, errors="replace").decode(enc)
        print(f"    {idx} -> {cls_s:<12} | train: {count_tr:>4} | test: {count_te:>4}")

    # ----------------------------------------------------------
    # STEP 3 - Load SBERT model
    # ----------------------------------------------------------
    print_section("LOAD SBERT MODEL")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Dimensi output: 384")
    print(f"  Bahasa: Multilingual (inkl. Indonesia)")
    print(f"  Memuat model...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: sentence-transformers belum terinstall.")
        print("  Jalankan: pip install sentence-transformers")
        return

    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model berhasil dimuat.")

    # ----------------------------------------------------------
    # STEP 4 - Encode train set
    # ----------------------------------------------------------
    print_section("ENCODE TRAIN SET")
    print(f"  Mengencode {len(X_train_text)} review...")
    print(f"  batch_size={BATCH_SIZE} | normalize_embeddings=True")

    X_train_emb = model.encode(
        X_train_text,
        batch_size          = BATCH_SIZE,
        show_progress_bar   = True,
        convert_to_numpy    = True,
        normalize_embeddings = True,   # L2 normalization -> unit vector
    )

    # [TAMBAHAN] Cast ke float32 — safeguard eksplisit + hemat memori disk
    # SBERT sudah output float32 by default, ini memastikan konsistensi
    # Catatan: sklearn SVC akan auto-cast ke float64 saat .fit() — normal
    X_train_emb = X_train_emb.astype(np.float32)

    print(f"  Shape: {X_train_emb.shape}")
    print(f"  dtype: {X_train_emb.dtype}")
    # Verifikasi L2 norm = 1.0 (sampel spot-check)
    norms = np.linalg.norm(X_train_emb[:5], axis=1)
    print(f"  L2 norm sample (seharusnya ~1.0): {norms.round(4).tolist()}")

    # ----------------------------------------------------------
    # STEP 5 - Encode test set
    # ----------------------------------------------------------
    print_section("ENCODE TEST SET")
    print(f"  Mengencode {len(X_test_text)} review...")

    X_test_emb = model.encode(
        X_test_text,
        batch_size           = BATCH_SIZE,
        show_progress_bar    = True,
        convert_to_numpy     = True,
        normalize_embeddings = True,   # Sama persis dengan train
    )

    # [TAMBAHAN] Cast ke float32 — konsisten dengan X_train_emb
    X_test_emb = X_test_emb.astype(np.float32)

    print(f"  Shape: {X_test_emb.shape}")
    print(f"  dtype: {X_test_emb.dtype}")   # [TAMBAHAN] tampilkan dtype untuk verifikasi

    # ----------------------------------------------------------
    # STEP 6 - Simpan ke .npy
    # ----------------------------------------------------------
    print_section("SIMPAN EMBEDDINGS")

    os.makedirs(DATA_DIR, exist_ok=True)

    np.save(X_TRAIN_EMB, X_train_emb)
    np.save(X_TEST_EMB,  X_test_emb)
    np.save(Y_TRAIN,     y_train_enc)
    np.save(Y_TEST,      y_test_enc)
    np.save(LABEL_CLS,   le.classes_)

    # Hitung ukuran file
    def mb(path):
        return os.path.getsize(path) / 1024 / 1024

    print(f"  X_train_emb.npy : {X_train_emb.shape}  -> {mb(X_TRAIN_EMB):.1f} MB")
    print(f"  X_test_emb.npy  : {X_test_emb.shape}   -> {mb(X_TEST_EMB):.1f} MB")
    print(f"  y_train.npy     : {y_train_enc.shape}")
    print(f"  y_test.npy      : {y_test_enc.shape}")
    print(f"  label_classes.npy: {le.classes_.tolist()}")

    # ----------------------------------------------------------
    # STEP 7 - Ringkasan & reminder
    # ----------------------------------------------------------
    print_section("RINGKASAN")
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Dimensi        : {X_train_emb.shape[1]}")
    print(f"  Normalisasi    : L2 (unit vector, norm=1.0)")
    print(f"  dtype          : {X_train_emb.dtype} (float32, hemat ~50% disk vs float64)")
    print(f"  Seed           : {SEED} (reproducible)")
    print(f"  Train embedding: {X_train_emb.shape}")
    print(f"  Test  embedding: {X_test_emb.shape}")

    print(f"\n{'='*60}")
    print(f"  CARA LOAD DI TAHAP MODELING")
    print(f"{'='*60}")
    print(f"  import numpy as np")
    print(f"  X_train = np.load('data/X_train_emb.npy')")
    print(f"  X_test  = np.load('data/X_test_emb.npy')")
    print(f"  y_train = np.load('data/y_train.npy')")
    print(f"  y_test  = np.load('data/y_test.npy')")
    print(f"  classes = np.load('data/label_classes.npy', allow_pickle=True)")
    print(f"  # classes = ['negative' 'neutral' 'positive']")

    print(f"\n{'='*60}")
    print(f"  TAHAP BERIKUTNYA — MODELING")
    print(f"{'='*60}")
    print(f"  1. Baseline dulu (tanpa SMOTE):")
    print(f"     SVC(kernel='linear', class_weight='balanced')")
    print(f"  2. Evaluasi: Accuracy, Precision, Recall, F1")
    print(f"     Metric utama: Macro F1 (neutral hanya ~5%)")
    print(f"  3. Jika Macro F1 masih rendah karena neutral:")
    print(f"     Coba SMOTE hanya pada X_train + y_train")
    print(f"     Bandingkan hasilnya dengan baseline")
    print(f"\n  JANGAN ENCODE ULANG — gunakan file .npy yang sudah ada.")
    print(f"  SELESAI")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
