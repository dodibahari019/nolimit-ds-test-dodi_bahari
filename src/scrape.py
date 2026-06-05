"""
GOOGLE PLAY STORE SCRAPER — ChatGPT Reviews
=============================================
TARGET APP  : ChatGPT (com.openai.chatgpt)

OUTPUT      : playstore_reviews_chatgpt.csv (satu file final)

KOLOM OUTPUT:
    review_id       — ID unik review dari Play Store
    review_text     — Teks review lengkap
    rating          — Rating bintang (1–5)
    review_date     — Tanggal review (YYYY-MM-DD HH:MM:SS)
    app_version     — Versi aplikasi saat review ditulis
    sentiment_label — Weak label: positive / neutral / negative
    word_count      — Jumlah kata

FILTER YANG DITERAPKAN:
    - Hanya bahasa Indonesia (langdetect)
    - Minimum 5 kata
    - Hapus review noise (emoji-only, tanda baca saja, dll.)
    - Hapus duplikat review_id

INSTALL:
    pip install google-play-scraper pandas tqdm langdetect
"""

import sys
import time
import os
import re
import pandas as pd
from tqdm import tqdm
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from google_play_scraper import reviews, Sort
from google_play_scraper.exceptions import NotFoundError

DetectorFactory.seed = 42  # Reproducible language detection

# ================================================================
# KONFIGURASI
# ================================================================

APP_ID         = "com.openai.chatgpt"
LANG           = "id"
COUNTRY        = "id"
TARGET_REVIEWS = 20000        # Naikkan setelah verifikasi awal (coba 500 dulu)
BATCH_SIZE     = 200
SORT_ORDER     = Sort.NEWEST
SLEEP_BETWEEN  = 2

MIN_WORDS      = 5           # Minimum jumlah token/kata
TARGET_LANG    = "id"        # Hanya bahasa Indonesia

CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PATH_RAW       = os.path.join(DATA_DIR, "raw_reviews.csv")
OUTPUT_FILE    = os.path.join(CURRENT_DIR, "playstore_reviews_chatgpt.csv")

# ================================================================
# HELPERS
# ================================================================

def weak_label(rating) -> str:
    try:
        r = int(rating)
        if r <= 2:   return "negative"
        elif r == 3: return "neutral"
        else:        return "positive"
    except (TypeError, ValueError):
        return "unknown"


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def is_valid_review(text: str) -> bool:
    """
    Validasi kualitas review:
      - Minimal MIN_WORDS kata
      - Minimal 3 huruf alfabet (bukan noise: emoji, angka, tanda baca saja)
    """
    if not text or len(text.strip()) == 0:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    if len(re.findall(r'[a-zA-Z]', text)) < 3:
        return False
    return True


def clean_text(text: str) -> str:
    """Hapus spasi berlebih saja. Stemming/stopword di tahap NLP terpisah."""
    return re.sub(r'\s+', ' ', text).strip()


def print_distribution(df: pd.DataFrame, col: str, title: str):
    if col not in df.columns or df.empty:
        return
    print(f"\n  {title}:")
    total = len(df)
    for val, count in df[col].value_counts().sort_index().items():
        print(f"    {str(val):<12} : {count:>5}  ({count/total*100:.1f}%)")


# ================================================================
# SCRAPING
# ================================================================

def scrape_playstore_reviews() -> pd.DataFrame:
    all_reviews  = []
    continuation = None
    batch_num    = 0
    no_new_count = 0

    print(f"\n{'='*60}")
    print(f"  Google Play Store Scraper")
    print(f"  App     : {APP_ID}")
    print(f"  Lang    : {LANG} | Country : {COUNTRY}")
    print(f"  Target  : {TARGET_REVIEWS} reviews")
    print(f"{'='*60}\n")

    with tqdm(total=TARGET_REVIEWS, desc="Scraping", unit="review") as pbar:
        while len(all_reviews) < TARGET_REVIEWS:
            try:
                batch_num += 1

                result, continuation = reviews(
                    APP_ID,
                    lang=LANG,
                    country=COUNTRY,
                    sort=SORT_ORDER,
                    count=BATCH_SIZE,
                    continuation_token=continuation,
                )

                if not result:
                    no_new_count += 1
                    print(f"\n  Batch {batch_num}: Tidak ada data baru ({no_new_count}x berturut-turut)")
                    if no_new_count >= 3:
                        print("  Berhenti: 3x berturut-turut kosong.")
                        break
                    time.sleep(SLEEP_BETWEEN * 2)
                    continue

                no_new_count = 0

                for r in result:
                    all_reviews.append({
                        "review_id":   r.get("reviewId", ""),
                        "review_text": (r.get("content") or "").strip(),
                        "rating":      r.get("score", None),
                        "review_date": r.get("at", None),
                        "app_version": r.get("appVersion", ""),
                    })

                pbar.update(len(result))
                print(
                    f"  Batch {batch_num:>3} | +{len(result):>3} | "
                    f"Total mentah: {len(all_reviews):>5}"
                )

                if len(all_reviews) >= TARGET_REVIEWS:
                    print("\n  Target tercapai!")
                    break

                if continuation is None:
                    print("\n  Tidak ada halaman berikutnya.")
                    break

                time.sleep(SLEEP_BETWEEN)

            except NotFoundError:
                print(f"\n  App '{APP_ID}' tidak ditemukan di Play Store.")
                break
            except Exception as e:
                print(f"\n  Error batch {batch_num}: {e}")
                print(f"  Retry dalam {SLEEP_BETWEEN * 3:.0f} detik...")
                time.sleep(SLEEP_BETWEEN * 3)
                continue

    return pd.DataFrame(all_reviews)


# ================================================================
# FILTER + LABEL + FINALIZE (SATU PIPELINE)
# ================================================================

def process(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    print(f"\n{'='*60}")
    print(f"  FILTER & LABELING PIPELINE")
    print(f"{'='*60}")
    print(f"  Input (mentah)                   : {len(df)}")

    # 1. Duplikat
    df = df.drop_duplicates(subset=["review_id"])
    print(f"  Setelah drop duplikat            : {len(df)}")

    # 2. Validasi teks (min kata + noise filter)
    df = df[df["review_text"].apply(is_valid_review)].copy()
    print(f"  Setelah filter min {MIN_WORDS} kata + noise : {len(df)}")

    # 3. Clean whitespace
    df["review_text"] = df["review_text"].apply(clean_text)

    # 4. Deteksi bahasa
    print(f"\n  Mendeteksi bahasa {len(df)} review...")
    languages = []
    for text in tqdm(df["review_text"], desc="  Deteksi bahasa", unit="review"):
        languages.append(detect_language(text))
    df["language"] = languages
    print_distribution(df, "language", "Distribusi bahasa sebelum filter")

    # 5. Filter bahasa Indonesia
    df = df[df["language"] == TARGET_LANG].copy()
    print(f"\n  Setelah filter bahasa '{TARGET_LANG}'    : {len(df)}")

    # 6. Hapus kolom 'language' — tidak masuk output final
    df = df.drop(columns=["language"])

    # 7. Weak label
    df["sentiment_label"] = df["rating"].apply(weak_label)

    # 8. Word count
    df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))

    # 9. Format tipe
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce") \
                          .dt.strftime("%Y-%m-%d %H:%M:%S")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df["review_text"] = df["review_text"].fillna("")
    df["app_version"] = df["app_version"].fillna("")

    # 10. Urutan kolom final
    col_order = [
        "review_id", "review_text", "rating", "review_date",
        "app_version", "sentiment_label", "word_count",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.reset_index(drop=True)

    return df


# ================================================================
# RINGKASAN
# ================================================================

def print_summary(n_raw: int, df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  SELESAI")
    print(f"{'='*60}")
    print(f"  Review mentah (raw)    : {n_raw}")
    print(f"  Review final           : {len(df)}")
    print(f"  Retention rate         : {len(df)/n_raw*100:.1f}%" if n_raw > 0 else "")
    print(f"  File output            : {OUTPUT_FILE}")

    if df.empty:
        return

    print_distribution(df, "rating",          "Distribusi Rating (bintang)")
    print_distribution(df, "sentiment_label", "Distribusi Sentiment Label")

    if "word_count" in df.columns:
        wc = df["word_count"]
        print(f"\n  Statistik Word Count:")
        print(f"    Min       : {wc.min()}")
        print(f"    Max       : {wc.max()}")
        print(f"    Rata-rata : {wc.mean():.1f}")
        print(f"    Median    : {wc.median():.0f}")

    print(f"\n  Sample review per sentiment:")
    print("-" * 60)
    for label in ["positive", "neutral", "negative"]:
        subset = df[df["sentiment_label"] == label]
        if subset.empty:
            continue
        sample = subset.iloc[len(subset) // 2]
        text   = str(sample["review_text"])
        # Encode safe untuk terminal Windows (cp1252) — ganti karakter tak dikenal
        text_safe = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(f"  [{label.upper()}] rating={sample.get('rating', '-')}")
        print(f"  {text_safe[:140]}{'...' if len(text_safe) > 140 else ''}")
        print()
    print("=" * 60)

    # Peringatan imbalance
    pos_pct = (df["sentiment_label"] == "positive").sum() / len(df) * 100
    if pos_pct > 80:
        print(
            f"\n  PERHATIAN: Dataset imbalanced ({pos_pct:.0f}% positive).\n"
            f"  Pertimbangkan oversampling (SMOTE) atau class weighting saat training.\n"
        )


# ================================================================
# MAIN
# ================================================================

def main():
    # Step 1 — Scraping
    df_raw = scrape_playstore_reviews()

    if df_raw.empty:
        print("\n  Tidak ada data. Periksa koneksi / app_id.")
        return

    n_raw = len(df_raw)

    # Step 2 — Simpan raw dataset
    df_raw_save = df_raw.copy()
    df_raw_save["review_date"] = pd.to_datetime(df_raw_save["review_date"], errors="coerce") \
                                    .dt.strftime("%Y-%m-%d %H:%M:%S")
    df_raw_save.to_csv(PATH_RAW, index=False, encoding="utf-8-sig")
    print(f"\n  [RAW] {len(df_raw_save)} review disimpan -> {PATH_RAW}")

    # Step 3 — Filter + Label (satu pipeline)
    df_final = process(df_raw)

    if df_final.empty:
        print("\n  Tidak ada review yang lolos filter.")
        return

    # Step 4 — Simpan file CSV final
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Step 5 — Ringkasan
    print_summary(n_raw, df_final)


if __name__ == "__main__":
    main()
