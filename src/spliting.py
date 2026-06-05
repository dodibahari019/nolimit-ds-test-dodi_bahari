"""
TRAIN-TEST SPLIT — ChatGPT Play Store Reviews
===============================================
INPUT  : preprocessed_reviews.csv
OUTPUT :
    data/train.csv   — 80% data untuk training
    data/test.csv    — 20% data untuk evaluasi

STRATEGI:
  - Stratified split (stratify=y) agar distribusi kelas
    positive/neutral/negative tetap proporsional di train & test
  - random_state=42 untuk reproducibility
  - Simpan ke file terpisah agar split tidak berubah tiap run

TIDAK dilakukan di sini:
  - SBERT embedding       (tahap berikutnya)
  - SMOTE / augmentasi    (setelah embedding, hanya pada train)
  - Training model        (setelah embedding)

INSTALL:
  pip install pandas scikit-learn
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ================================================================
# KONFIGURASI
# ================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(CURRENT_DIR, "preprocessed_reviews.csv")
DATA_DIR    = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_FILE  = os.path.join(DATA_DIR, "train.csv")
TEST_FILE   = os.path.join(DATA_DIR, "test.csv")

TEST_SIZE    = 0.2    # 80% train, 20% test
RANDOM_STATE = 42     # Seed untuk reproducibility
LABEL_COL    = "sentiment_label"
TEXT_COL     = "review_text"

# ================================================================
# HELPERS
# ================================================================

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_dist(label: str, series: pd.Series):
    """Print distribusi kelas dengan persentase."""
    print(f"\n  {label}:")
    total = len(series)
    for cls in sorted(series.unique()):
        count = (series == cls).sum()
        enc   = sys.stdout.encoding or "utf-8"
        cls_s = str(cls).encode(enc, errors="replace").decode(enc)
        print(f"    {cls_s:<12} : {count:>5}  ({count/total*100:.1f}%)")


def check_min_samples(y: pd.Series, test_size: float) -> bool:
    """
    Verifikasi setiap kelas punya cukup sampel untuk di-split.
    Stratified split butuh minimal 2 sampel per kelas
    (1 untuk train, 1 untuk test).
    Untuk test_size=0.2, butuh minimal ~5 sampel per kelas agar
    ada >= 1 di test set.
    """
    min_needed = max(2, int(1 / test_size) + 1)
    counts     = y.value_counts()
    ok         = True
    for cls, count in counts.items():
        if count < min_needed:
            enc   = sys.stdout.encoding or "utf-8"
            cls_s = str(cls).encode(enc, errors="replace").decode(enc)
            print(f"  PERINGATAN: Kelas '{cls_s}' hanya {count} sampel "
                  f"(butuh minimal {min_needed} untuk stratified split).")
            ok = False
    return ok


# ================================================================
# MAIN
# ================================================================

def main():

    # ----------------------------------------------------------
    # STEP 1 - Load dataset
    # ----------------------------------------------------------
    print_section("LOAD DATASET")

    if not os.path.exists(INPUT_FILE):
        print(f"  ERROR: File tidak ditemukan: {INPUT_FILE}")
        print(f"  Pastikan preprocessing.py sudah dijalankan terlebih dahulu.")
        return

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"  File input  : {INPUT_FILE}")
    print(f"  Total review: {len(df)}")

    # Cek kolom yang diperlukan ada
    for col in [TEXT_COL, LABEL_COL]:
        if col not in df.columns:
            print(f"  ERROR: Kolom '{col}' tidak ditemukan di dataset.")
            return

    # ----------------------------------------------------------
    # STEP 2 - Audit distribusi kelas sebelum split
    # ----------------------------------------------------------
    print_section("DISTRIBUSI KELAS SEBELUM SPLIT")
    print_dist("Keseluruhan dataset", df[LABEL_COL])

    # Cek apakah cukup sampel untuk stratified split
    print()
    check_min_samples(df[LABEL_COL], TEST_SIZE)

    # ----------------------------------------------------------
    # STEP 3 - Train-Test Split (stratified)
    # ----------------------------------------------------------
    print_section("TRAIN-TEST SPLIT")
    print(f"  Rasio       : {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test")
    print(f"  Stratified  : Ya (stratify=sentiment_label)")
    print(f"  Random state: {RANDOM_STATE}")

    df_train, df_test = train_test_split(
        df,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = df[LABEL_COL],
        shuffle      = True,
    )

    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    # ----------------------------------------------------------
    # STEP 4 - Verifikasi distribusi setelah split
    # ----------------------------------------------------------
    print_section("VERIFIKASI DISTRIBUSI SETELAH SPLIT")
    print_dist(f"Train set  ({len(df_train)} sampel)", df_train[LABEL_COL])
    print_dist(f"Test set   ({len(df_test)} sampel)",  df_test[LABEL_COL])

    # Cek konsistensi distribusi train vs test
    print(f"\n  Konsistensi distribusi (train vs test vs full):")
    print(f"    {'Kelas':<12}  {'Full':>8}  {'Train':>8}  {'Test':>8}")
    print(f"    {'-'*44}")
    for cls in sorted(df[LABEL_COL].unique()):
        enc     = sys.stdout.encoding or "utf-8"
        cls_s   = str(cls).encode(enc, errors="replace").decode(enc)
        pct_all = (df[LABEL_COL] == cls).sum() / len(df) * 100
        pct_tr  = (df_train[LABEL_COL] == cls).sum() / len(df_train) * 100
        pct_te  = (df_test[LABEL_COL] == cls).sum() / len(df_test) * 100
        print(f"    {cls_s:<12}  {pct_all:>7.1f}%  {pct_tr:>7.1f}%  {pct_te:>7.1f}%")

    # ----------------------------------------------------------
    # STEP 5 - Simpan ke CSV
    # ----------------------------------------------------------
    print_section("SIMPAN HASIL SPLIT")

    df_train.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_FILE,  index=False, encoding="utf-8-sig")

    print(f"  train.csv   : {len(df_train)} review -> {TRAIN_FILE}")
    print(f"  test.csv    : {len(df_test)} review  -> {TEST_FILE}")
    print(f"\n  Kolom yang tersimpan:")
    for col in df_train.columns:
        print(f"    - {col}")

    # ----------------------------------------------------------
    # STEP 6 - Ringkasan & reminder tahap berikutnya
    # ----------------------------------------------------------
    print_section("RINGKASAN")
    print(f"  Total dataset          : {len(df)}")
    print(f"  Train set              : {len(df_train)}  ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Test set               : {len(df_test)}   ({len(df_test)/len(df)*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  TAHAP BERIKUTNYA")
    print(f"{'='*60}")
    print(f"  Input untuk SBERT embedding:")
    print(f"    X_train = train.csv['review_text']")
    print(f"    X_test  = test.csv['review_text']")
    print(f"    y_train = train.csv['sentiment_label']")
    print(f"    y_test  = test.csv['sentiment_label']")
    print(f"\n  Urutan pipeline setelah ini:")
    print(f"    1. SBERT Embedding")
    print(f"       X_train -> encode() -> X_train_emb  (shape: {len(df_train)} x 768)")
    print(f"       X_test  -> encode() -> X_test_emb   (shape: {len(df_test)} x 768)")
    print(f"    2. Handling imbalanced (HANYA pada train set)")
    print(f"       Baseline dulu: SVC(class_weight='balanced')")
    print(f"       Lalu bandingkan dengan SMOTE jika perlu")
    print(f"    3. Training SVM")
    print(f"    4. Evaluasi pada X_test_emb")
    print(f"       Metric utama: Macro F1 (karena neutral hanya ~5%)")
    print(f"\n  SELESAI")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
