"""
PREPROCESSING — ChatGPT Play Store Reviews
============================================
INPUT  : playstore_reviews_chatgpt.csv  (hasil scraping)
OUTPUT : preprocessed_reviews.csv       (siap masuk SBERT pipeline)

TAHAPAN PREPROCESSING:
  1. Load & audit dataset awal
  2. Drop duplikat review_text (bukan hanya review_id)
  3. Case folding (lowercase)
  4. Remove URL
  5. Remove mention (@username)
  6. Remove mojibake / replacement character (\ufffd)
  7. Normalisasi karakter berulang (bagussss → bagus)
  8. Normalize whitespace
  9. Drop review yang jadi kosong setelah cleaning
 10. Audit & simpan hasil

YANG TIDAK DILAKUKAN (karena pakai SBERT):
  ✗ Tokenisasi manual
  ✗ Stopword removal
  ✗ Stemming / lemmatisasi
  ✗ Hapus seluruh non-ASCII (terlalu agresif)
  ✗ Slang normalization

INSTALL:
  pip install pandas
"""

import os
import re
import sys
import pandas as pd

# ================================================================
# KONFIGURASI
# ================================================================

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(CURRENT_DIR, "playstore_reviews_chatgpt.csv")
OUTPUT_FILE  = os.path.join(CURRENT_DIR, "preprocessed_reviews.csv")

# ================================================================
# FUNGSI CLEANING
# ================================================================

def remove_urls(text: str) -> str:
    """Hapus URL (http, https, www)."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_mentions(text: str) -> str:
    """Hapus mention (@username)."""
    return re.sub(r"@\w+", " ", text)


def remove_mojibake(text: str) -> str:
    """
    Hapus replacement character (U+FFFD = '▯' / '?') dan karakter
    control (kecuali tab dan newline yang nanti dinormalisasi).
    TIDAK menghapus semua non-ASCII — terlalu agresif untuk teks Indonesia.
    """
    # Hapus U+FFFD (mojibake / encoding error)
    text = text.replace("\ufffd", " ")
    # Hapus karakter control (C0 & C1) kecuali \t dan \n
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    return text


def normalize_repeated_chars(text: str) -> str:
    """
    Normalisasi huruf (a-zA-Z) yang berulang lebih dari 2x berturut-turut.
    Contoh: 'bagussss' → 'baguss', 'mantappppp' → 'mantapp'
    HANYA huruf — emoji seperti 😍😍😍😍 dibiarkan utuh karena
    mengandung sinyal sentimen yang berguna untuk SBERT.
    """
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)


def normalize_whitespace(text: str) -> str:
    """Ganti semua whitespace berlebih (spasi, tab, newline) jadi satu spasi."""
    return re.sub(r"\s+", " ", text).strip()


def clean_pipeline(text: str) -> str:
    """
    Pipeline cleaning lengkap — urutan PENTING:
      1. Mojibake dulu sebelum lowercase (karakter kontrol tidak case-sensitif)
      2. Lowercase
      3. URL & mention
      4. Repeated chars
      5. Whitespace
    """
    if not isinstance(text, str):
        return ""
    text = remove_mojibake(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_repeated_chars(text)
    text = normalize_whitespace(text)
    return text


# ================================================================
# AUDIT HELPER
# ================================================================

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_stat(label: str, value, total: int = None):
    if total and isinstance(value, int):
        pct = value / total * 100 if total > 0 else 0
        print(f"  {label:<40} : {value:>6}  ({pct:.1f}%)")
    else:
        print(f"  {label:<40} : {value}")


def safe_print(text: str, max_len: int = 120):
    """Print aman untuk terminal Windows cp1252."""
    enc = sys.stdout.encoding or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc)
    print(f"  {safe[:max_len]}{'...' if len(safe) > max_len else ''}")


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():

    # ----------------------------------------------------------
    # STEP 1 — Load dataset
    # ----------------------------------------------------------
    print_section("LOAD DATASET")

    if not os.path.exists(INPUT_FILE):
        print(f"  ERROR: File tidak ditemukan: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    n_initial = len(df)
    print_stat("File input", INPUT_FILE)
    print_stat("Total review dimuat", n_initial)
    print_stat("Kolom", list(df.columns))

    # ----------------------------------------------------------
    # STEP 2 — Drop duplikat review_text
    # (selain duplikat review_id yang sudah dilakukan di scraping)
    # ----------------------------------------------------------
    print_section("DROP DUPLIKAT TEKS")

    n_before = len(df)
    df = df.drop_duplicates(subset=["review_text"]).copy()
    n_dedup = n_before - len(df)

    print_stat("Sebelum drop duplikat teks", n_before)
    print_stat("Duplikat teks dihapus", n_dedup, n_before)
    print_stat("Setelah drop duplikat teks", len(df))

    # ----------------------------------------------------------
    # STEP 3–8 — Cleaning pipeline
    # ----------------------------------------------------------
    print_section("CLEANING PIPELINE")
    print("  Langkah  : mojibake -> lowercase -> url -> mention -> repeated chars -> whitespace")

    # Simpan teks asli untuk perbandingan audit
    df["review_text_raw"] = df["review_text"].copy()

    # Terapkan pipeline
    df["review_text"] = df["review_text"].apply(clean_pipeline)

    # Hitung berapa review yang mengalami perubahan
    changed = (df["review_text"] != df["review_text_raw"]).sum()
    print_stat("Review yang mengalami perubahan", changed, len(df))

    # ----------------------------------------------------------
    # STEP 9 — Drop review kosong setelah cleaning
    # ----------------------------------------------------------
    print_section("DROP REVIEW KOSONG POST-CLEANING")

    n_before = len(df)
    # Juga drop yang kurang dari 3 kata setelah cleaning
    df = df[df["review_text"].str.strip().str.len() > 0].copy()
    df = df[df["review_text"].apply(lambda x: len(x.split()) >= 3)].copy()
    n_dropped = n_before - len(df)

    print_stat("Review kosong/terlalu pendek setelah cleaning", n_dropped, n_before)
    print_stat("Review tersisa", len(df))

    # Drop duplikat review_text LAGI setelah cleaning
    # Kasus: 'Bagusssss', 'BAGUSSSS', 'baguss' → setelah cleaning → 'baguss' (duplikat)
    n_before_dedup2 = len(df)
    df = df.drop_duplicates(subset=["review_text"]).copy()
    n_dedup2 = n_before_dedup2 - len(df)
    print_stat("Duplikat teks post-cleaning dihapus", n_dedup2, n_before_dedup2)
    print_stat("Review tersisa setelah dedup kedua", len(df))

    # Hapus kolom review_text_raw (tidak masuk output)
    df = df.drop(columns=["review_text_raw"])

    # Update word_count setelah cleaning
    df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))

    # Reset index
    df = df.reset_index(drop=True)

    # ----------------------------------------------------------
    # STEP 10 — Audit hasil + simpan
    # ----------------------------------------------------------
    print_section("AUDIT HASIL PREPROCESSING")

    print_stat("Total review awal (raw)", n_initial)
    print_stat("Total review final (clean)", len(df))
    print_stat("Total dihapus", n_initial - len(df), n_initial)

    # Distribusi sentiment
    print(f"\n  Distribusi Sentiment Label:")
    total = len(df)
    for label, count in df["sentiment_label"].value_counts().sort_index().items():
        enc = sys.stdout.encoding or "utf-8"
        label_safe = str(label).encode(enc, errors="replace").decode(enc)
        print(f"    {label_safe:<12} : {count:>5}  ({count/total*100:.1f}%)")

    # Statistik word count setelah cleaning
    wc = df["word_count"]
    print(f"\n  Statistik Word Count (post-cleaning):")
    print(f"    Min       : {wc.min()}")
    print(f"    Max       : {wc.max()}")
    print(f"    Rata-rata : {wc.mean():.1f}")
    print(f"    Median    : {wc.median():.0f}")

    # Contoh sebelum/sesudah cleaning (ambil yang berubah)
    df_temp = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df_temp = df_temp.drop_duplicates(subset=["review_text"]).copy()
    df_temp["cleaned"] = df_temp["review_text"].apply(clean_pipeline)
    changed_examples = df_temp[df_temp["review_text"] != df_temp["cleaned"]].head(5)

    if len(changed_examples) > 0:
        print(f"\n  Contoh perubahan (before -> after):")
        print("-" * 60)
        for _, row in changed_examples.iterrows():
            print("  BEFORE:")
            safe_print(str(row["review_text"]))
            print("  AFTER :")
            safe_print(str(row["cleaned"]))
            print()

    # Simpan
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  File output: {OUTPUT_FILE}")

    # ----------------------------------------------------------
    # PERINGATAN IMBALANCE (reminder untuk tahap berikutnya)
    # ----------------------------------------------------------
    pos_pct = (df["sentiment_label"] == "positive").sum() / total * 100
    neu_pct = (df["sentiment_label"] == "neutral").sum()  / total * 100
    neg_pct = (df["sentiment_label"] == "negative").sum() / total * 100

    print(f"\n{'='*60}")
    print(f"  REMINDER UNTUK TAHAP SELANJUTNYA")
    print(f"{'='*60}")
    print(f"  Urutan pipeline yang benar setelah ini:")
    print(f"    1. Train-Test Split (stratified, 80:20)")
    print(f"    2. SBERT Embedding (train & test terpisah)")
    print(f"    3. Handling imbalanced HANYA pada train set")
    print(f"       (class_weight='balanced' atau SMOTE pada vector)")
    print(f"    4. Training model")
    print(f"    5. Evaluasi pada test set")

    if pos_pct > 70:
        print(f"\n  PERHATIAN: Imbalanced dataset!")
        print(f"    Positive : {pos_pct:.1f}%")
        print(f"    Neutral  : {neu_pct:.1f}%")
        print(f"    Negative : {neg_pct:.1f}%")
        print(f"    -> Gunakan stratified split + class_weight='balanced'")
        print(f"    -> Neutral ({neu_pct:.1f}%) sangat sedikit - pertimbangkan")
        print(f"      apakah merge ke negative atau tetap 3 kelas")

    print(f"\n  SELESAI")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
