"""
=============================================================================
PREDICT -- Inferensi / Prediksi Sentimen Review Baru
=============================================================================
Proyek  : Sentiment Analysis Review ChatGPT Play Store (Indonesian)
Tahap   : Inferensi (setelah Preprocessing, Embedding, Modeling, Evaluasi)
Versi   : v1.0

=============================================================================
POSISI SCRIPT INI DALAM PIPELINE LENGKAP
=============================================================================

  [1] preprocessing.py      -- Cleaning & dedup data mentah
          |
          v
  [2] splitting.py          -- Train-test split stratified 80:20
          |
          v
  [3] embedding.py -- SBERT embedding -> .npy
          |
          v
  [4] modeling.py           -- SVM + GridSearchCV + CV tuning
          |
          v
  [5] evaluasi.py           -- Evaluasi final pada test set
          |
          v
  [6] predict.py            -- ANDA DI SINI
          |
          v
       output/sample_predictions.csv

  Script ini TIDAK melakukan training, tuning, CV, evaluasi, atau GridSearch.
  Semua proses tersebut sudah selesai di tahap sebelumnya.
  Script ini hanya: preprocessing -> embedding -> load model -> prediksi.

=============================================================================
DESKRIPSI TEKNIS
=============================================================================

  PREPROCESSING (identik dengan preprocessing.py):
    - remove_mojibake       : hapus U+FFFD dan karakter control
    - lowercase             : case folding
    - remove_urls           : hapus http/https/www
    - remove_mentions       : hapus @username
    - normalize_repeated    : bagussss -> baguss
    - normalize_whitespace  : collapse multiple spaces

    YANG TIDAK DILAKUKAN (sama seperti training -- konsistensi wajib):
    [X] Tokenisasi manual
    [X] Stopword removal
    [X] Stemming / lemmatisasi
    [X] Hapus seluruh non-ASCII
    [X] Slang normalization

  EMBEDDING (identik dengan embedding.py):
    - Model  : paraphrase-multilingual-MiniLM-L12-v2
    - Dimensi: 384
    - L2 normalization (normalize_embeddings=True)
    - batch_size=32
    - dtype: float32

    ALASAN KONSISTENSI WAJIB:
    Jika preprocessing atau embedding parameter berbeda dari training,
    embedding vector yang dihasilkan akan berada di ruang yang berbeda
    dari hyperplane yang dipelajari SVM -> prediksi salah total.

  MODEL (hasil tahap modeling.py + hyperparameter tuning):
    - Algoritma   : SVC (C-Support Vector Classification)
    - Kernel      : RBF (Radial Basis Function)
    - C           : 1
    - gamma       : scale (= 1 / (n_features * X.var()))
    - class_weight: balanced
    - probability : True (Platt Scaling -> predict_proba tersedia)
    - CV F1-Macro (training) : 58.84%
    - Test F1-Macro (evaluasi): 60.06%
    - Test AUC-Macro          : 0.8717

  LABEL MAPPING (dari label_classes.npy -- hasil LabelEncoder training):
    0 -> negative
    1 -> neutral
    2 -> positive

  OUTPUT KOLOM:
    id               : nomor urut prediksi (1, 2, 3, ...)
    review_original  : teks review asli (sebelum cleaning)
    review_clean     : teks setelah preprocessing
    word_count       : jumlah kata setelah cleaning
    prediction       : label sentimen hasil prediksi
    confidence       : max probability (keyakinan model terhadap label)
    conf_negative    : probabilitas kelas negative
    conf_neutral     : probabilitas kelas neutral
    conf_positive    : probabilitas kelas positive
    low_confidence   : True jika confidence < threshold (default 0.50)
    short_review     : True jika panjang teks < 3 kata
    prediction_time  : timestamp saat prediksi dijalankan

=============================================================================
INPUT:
  data/sample_data.csv               -- CSV dengan kolom 'review'
  data/label_classes.npy             -- Label encoder dari tahap training
  output/svm_sbert_best_model.joblib -- Model terbaik dari tahap Modeling

OUTPUT:
  output/sample_predictions.csv      -- Prediksi lengkap dengan confidence

INSTALL:
  pip install sentence-transformers scikit-learn pandas numpy joblib colorama
=============================================================================
"""

import os
import re
import sys
import time
import numpy as np
import pandas as pd
import joblib

# =============================================================================
# KONFIGURASI
# =============================================================================

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(CURRENT_DIR, "data")
OUTPUT_DIR   = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input
IN_SAMPLE_CSV = os.path.join(DATA_DIR,   "sample_data.csv")
IN_BEST_MODEL = os.path.join(OUTPUT_DIR, "svm_sbert_best_model.joblib")
IN_LABEL_CLS  = os.path.join(DATA_DIR,   "label_classes.npy")

# Output
OUT_PREDICT   = os.path.join(OUTPUT_DIR, "sample_predictions.csv")

# SBERT -- HARUS IDENTIK dengan embedding.py
# Mengganti model ini akan menghasilkan embedding di ruang vektor berbeda
# sehingga hyperplane SVM tidak relevan lagi dan prediksi menjadi salah.
SBERT_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
SBERT_DIM     = 384
BATCH_SIZE    = 32     # Aman untuk CPU maupun GPU RAM 8GB+

# Kolom teks di CSV input
TEXT_COL      = "review"

# Threshold confidence untuk flagging prediksi yang tidak yakin.
# Review dengan max_prob < threshold -> low_confidence = True.
# Bisa digunakan untuk routing ke human reviewer di produksi.
CONF_THRESHOLD = 0.50

# Minimum panjang teks (kata) agar prediksi dianggap reliable
MIN_WORDS     = 3

# =============================================================================
# LOGGING HELPERS
# =============================================================================
try:
    import colorama
    colorama.init(autoreset=True)
    C_OK   = colorama.Fore.GREEN
    C_WARN = colorama.Fore.YELLOW
    C_ERR  = colorama.Fore.RED
    C_HEAD = colorama.Fore.CYAN + colorama.Style.BRIGHT
    C_RST  = colorama.Style.RESET_ALL
except ImportError:
    C_OK = C_WARN = C_ERR = C_HEAD = C_RST = ""


def log_ok(msg):
    print(C_OK + "  [OK]   " + str(msg) + C_RST)


def log_warn(msg):
    print(C_WARN + "  [WARN] " + str(msg) + C_RST)


def log_err(msg):
    print(C_ERR + "  [ERR]  " + str(msg) + C_RST)


def print_section(title):
    line = "=" * 65
    print("\n" + line)
    print("  " + title)
    print(line)


def safe_print(text, max_len=120):
    """Print aman untuk terminal dengan encoding terbatas (mis. Windows cp1252)."""
    enc  = sys.stdout.encoding or "utf-8"
    safe = str(text).encode(enc, errors="replace").decode(enc)
    if len(safe) > max_len:
        print("  " + safe[:max_len] + "...")
    else:
        print("  " + safe)


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
# PENTING: Fungsi-fungsi ini HARUS identik dengan preprocessing.py
# yang digunakan saat training. Perbedaan sekecil apapun dalam
# preprocessing akan membuat embedding berbeda -> prediksi tidak valid.
#
# Checklist konsistensi dengan preprocessing.py:
#   [OK] remove_mojibake           -- hapus U+FFFD & karakter control
#   [OK] lowercase                 -- case folding
#   [OK] remove_urls               -- http/https/www
#   [OK] remove_mentions           -- @username
#   [OK] normalize_repeated_chars  -- hanya huruf a-zA-Z, bukan emoji
#   [OK] normalize_whitespace      -- collapse + strip
#
# Yang TIDAK dilakukan (konsisten dengan keputusan training):
#   [X] Hapus emoji      -- emoji = sinyal sentimen untuk SBERT
#   [X] Stopword removal -- SBERT memproses konteks penuh
#   [X] Stemming         -- SBERT sudah menangani morfologi
# =============================================================================

def remove_urls(text):
    """Hapus URL (http, https, www)."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_mentions(text):
    """Hapus mention (@username)."""
    return re.sub(r"@\w+", " ", text)


def remove_mojibake(text):
    """
    Hapus replacement character (U+FFFD) dan karakter control.
    Tidak menghapus semua non-ASCII karena terlalu agresif untuk teks
    Indonesia yang mengandung karakter latin beraksara dan emoji.
    """
    text = text.replace("\ufffd", " ")
    # Hapus karakter control (C0 & C1) kecuali tab dan newline
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    return text


def normalize_repeated_chars(text):
    """
    Normalisasi huruf (a-zA-Z) yang berulang lebih dari 2x berturut-turut.
    Contoh: bagussss -> baguss | mantappppp -> mantapp

    HANYA huruf -- emoji dibiarkan utuh karena mengandung sinyal sentimen
    yang berguna untuk SBERT. Konsisten dengan preprocessing.py.
    """
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)


def normalize_whitespace(text):
    """Ganti semua whitespace berlebih (spasi, tab, newline) jadi satu spasi."""
    return re.sub(r"\s+", " ", text).strip()


def clean_pipeline(text):
    """
    Pipeline cleaning lengkap -- urutan PENTING dan IDENTIK dengan preprocessing.py.

    Urutan langkah:
      1. remove_mojibake
         Dilakukan pertama, sebelum lowercase, karena karakter control
         tidak case-sensitif dan perlu dibersihkan lebih dulu.

      2. lowercase
         Case folding setelah mojibake agar tidak ada UPPERCASE yang
         tersisa dari karakter yang sudah dibersihkan.

      3. remove_urls
         URL dihapus setelah lowercase agar pattern match lebih konsisten.
         HTTP:// sudah jadi http:// setelah lowercase.

      4. remove_mentions
         @username dihapus, tidak relevan untuk analisis sentimen.

      5. normalize_repeated_chars
         Hanya huruf, dilakukan setelah lowercase agar BAGUSSSS
         (sudah jadi bagussss) bisa di-normalize.

      6. normalize_whitespace
         Terakhir, setelah semua penghapusan yang mungkin meninggalkan
         multiple spaces, tab, atau newline.

    Returns:
        str: Teks yang sudah bersih, atau string kosong jika input bukan string.
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


# =============================================================================
# MAIN PREDICT PIPELINE
# =============================================================================

def main():
    t_start   = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    banner = (
        "\n" + "=" * 65 + "\n" +
        "  PREDICT -- Sentimen Review Indonesian\n" +
        "  Start      : " + timestamp + "\n" +
        "  Input      : data/sample_data.csv\n" +
        "  Labels     : data/label_classes.npy\n" +
        "  Output     : output/sample_predictions.csv\n" +
        "\n  MODEL YANG DIGUNAKAN:\n" +
        "    Algoritma : SVC -- RBF Kernel\n" +
        "    C         : 1  |  gamma: scale\n" +
        "    Embedding : SBERT " + SBERT_MODEL + "\n" +
        "    Dimensi   : " + str(SBERT_DIM) + "  |  L2-normalized\n" +
        "    CV F1-Macro (training) : 58.84%\n" +
        "    Test F1-Macro (evaluasi): 60.06%\n" +
        "    Test AUC-Macro         : 0.8717\n" +
        "\n  PIPELINE:\n" +
        "    CSV -> Preprocessing -> SBERT -> SVM -> Predictions\n" +
        "=" * 65
    )
    print(C_HEAD + banner + C_RST)

    # =========================================================================
    # STEP 1 -- VERIFIKASI FILE INPUT
    # =========================================================================
    print_section("STEP 1: VERIFIKASI FILE INPUT")

    required_files = [
        (IN_SAMPLE_CSV, "data/sample_data.csv",
         "Buat file dengan kolom 'review' berisi teks yang ingin diprediksi."),
        (IN_BEST_MODEL, "output/svm_sbert_best_model.joblib",
         "Pastikan modeling.py sudah dijalankan dan menghasilkan file ini."),
        (IN_LABEL_CLS,  "data/label_classes.npy",
         "Pastikan embedding.py sudah dijalankan dan menghasilkan file ini."),
    ]

    all_ok = True
    for path, name, hint in required_files:
        if not os.path.exists(path):
            log_err("File tidak ditemukan: " + path)
            log_err("Petunjuk: " + hint)
            all_ok = False
        else:
            size_kb = os.path.getsize(path) // 1024
            log_ok(name + " (" + str(size_kb) + " KB)")

    if not all_ok:
        return

    # =========================================================================
    # STEP 2 -- LOAD DATA BARU
    # =========================================================================
    print_section("STEP 2: LOAD SAMPLE DATA")

    df = pd.read_csv(IN_SAMPLE_CSV, encoding="utf-8-sig")
    n_raw = len(df)

    print("  File    : " + IN_SAMPLE_CSV)
    print("  Kolom   : " + str(list(df.columns)))
    print("  Jumlah  : " + str(n_raw) + " baris")

    # Validasi kolom wajib
    if TEXT_COL not in df.columns:
        log_err("Kolom '" + TEXT_COL + "' tidak ditemukan.")
        log_err("Kolom yang tersedia: " + str(list(df.columns)))
        log_err("Pastikan CSV memiliki kolom bernama '" + TEXT_COL + "'.")
        return

    # Validasi ada data
    if n_raw == 0:
        log_err("File CSV kosong -- tidak ada data untuk diprediksi.")
        return

    # Simpan teks original (sebelum cleaning) untuk kolom output
    df["review_original"] = df[TEXT_COL].copy()

    log_ok(str(n_raw) + " review siap diproses")

    # =========================================================================
    # STEP 3 -- LOAD LABEL CLASSES
    # =========================================================================
    # Label classes dimuat dari label_classes.npy yang dihasilkan oleh
    # embedding.py saat training.
    # Format: array ['negative' 'neutral' 'positive']
    # Index array sesuai dengan output model.predict():
    #   index 0 -> negative
    #   index 1 -> neutral
    #   index 2 -> positive
    #
    # Cara ini lebih aman daripada hardcode LABEL_MAP karena:
    # - Urutan kelas dijamin konsisten dengan LabelEncoder saat training
    # - Jika kelas berubah di masa depan, file ini yang jadi sumber kebenaran
    # =========================================================================
    print_section("STEP 3: LOAD LABEL CLASSES")

    label_classes = np.load(IN_LABEL_CLS, allow_pickle=True)
    print("  File   : " + IN_LABEL_CLS)
    print("  Kelas  : " + str(label_classes.tolist()))
    print("  Jumlah : " + str(len(label_classes)) + " kelas")

    # Bangun LABEL_MAP dari label_classes.npy -- bukan hardcode
    # Key: integer index (0, 1, 2) | Value: nama kelas ('negative', dst)
    LABEL_MAP = {i: str(cls) for i, cls in enumerate(label_classes)}
    print("  Mapping: " + str(LABEL_MAP))

    log_ok("Label classes dimuat dari file (bukan hardcode)")

    # =========================================================================
    # STEP 4 -- PREPROCESSING
    # =========================================================================
    # Preprocessing HARUS identik dengan yang digunakan saat training.
    # Fungsi clean_pipeline() di sini adalah salinan langsung dari
    # preprocessing.py agar script ini bisa berdiri sendiri (standalone).
    #
    # Urutan tahap (sama persis dengan preprocessing.py):
    #   remove_mojibake -> lowercase -> remove_urls -> remove_mentions
    #   -> normalize_repeated_chars -> normalize_whitespace
    # =========================================================================
    print_section("STEP 4: PREPROCESSING")
    print("  Menerapkan cleaning pipeline (identik dengan preprocessing.py)...")
    print("  Tahap: mojibake -> lowercase -> url -> mention -> repeated -> whitespace")

    # Terapkan pipeline ke seluruh kolom review
    df["review_clean"] = df[TEXT_COL].apply(clean_pipeline)

    # Hitung word count setelah cleaning
    df["word_count"] = df["review_clean"].apply(lambda x: len(x.split()))

    # -- Audit hasil cleaning -----------------------------------------------

    # Review yang berubah setelah cleaning
    n_changed = (df["review_clean"] != df["review_original"].str.lower()).sum()

    # Review kosong setelah cleaning
    mask_empty = df["review_clean"].str.strip().str.len() == 0
    n_empty    = mask_empty.sum()

    # Review sangat pendek (< MIN_WORDS kata) tapi tidak kosong
    mask_short = (df["word_count"] < MIN_WORDS) & (~mask_empty)
    n_short    = mask_short.sum()

    print("\n  Audit Preprocessing:")
    print("    Total input               : " + str(n_raw))
    print("    Review yang dimodifikasi  : " + str(n_changed))
    print("    Review kosong post-clean  : " + str(n_empty))
    print("    Review < " + str(MIN_WORDS) + " kata (warning)  : " + str(n_short))

    if n_empty > 0:
        log_warn(str(n_empty) + " review kosong setelah cleaning.")
        log_warn("Review ini tetap diproses tapi prediksinya mungkin tidak akurat.")
        # Isi placeholder agar tidak crash saat embedding
        df.loc[mask_empty, "review_clean"] = "[empty]"

    if n_short > 0:
        log_warn(str(n_short) + " review sangat pendek (< " + str(MIN_WORDS) + " kata).")
        log_warn("Prediksi untuk review pendek umumnya kurang reliable.")

    # Contoh hasil cleaning -- ambil yang ada perubahan, maks 3
    df_changed = df[df["review_clean"] != df["review_original"].str.lower()].head(3)
    if len(df_changed) > 0:
        print("\n  Contoh perubahan cleaning:")
        for _, row in df_changed.iterrows():
            enc = sys.stdout.encoding or "utf-8"
            before = str(row["review_original"])[:70]
            after  = str(row["review_clean"])[:70]
            print(("    BEFORE: " + before).encode(enc, errors="replace").decode(enc))
            print(("    AFTER : " + after).encode(enc, errors="replace").decode(enc))
            print()

    log_ok("Preprocessing selesai")

    # =========================================================================
    # STEP 5 -- LOAD SBERT MODEL
    # =========================================================================
    # SBERT model HARUS identik dengan yang digunakan di embedding.py.
    # Parameter yang kritis untuk konsistensi:
    #   - Nama model         : paraphrase-multilingual-MiniLM-L12-v2
    #   - normalize_embeddings=True (L2 normalization ke unit vector)
    #   - batch_size         : tidak mempengaruhi hasil, hanya kecepatan
    #
    # Alasan model ini dipilih (dari keputusan di embedding.py):
    #   - Support Bahasa Indonesia
    #   - Ringan dan cepat (384-dim vs 768-dim)
    #   - Digunakan luas di riset NLP Indonesia
    # =========================================================================
    print_section("STEP 5: LOAD SBERT MODEL")
    print("  Model      : " + SBERT_MODEL)
    print("  Dimensi    : " + str(SBERT_DIM))
    print("  Bahasa     : Multilingual (inkl. Indonesia)")
    print("  Normalisasi: L2 (normalize_embeddings=True)")
    print("  Catatan    : Model HARUS sama persis dengan embedding.py")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log_err("Package sentence-transformers belum terinstall.")
        log_err("Jalankan: pip install sentence-transformers")
        return

    print("  Memuat model SBERT...")
    t0    = time.time()
    sbert = SentenceTransformer(SBERT_MODEL)
    log_ok("SBERT model dimuat dalam " + str(round(time.time() - t0, 2)) + "s")

    # =========================================================================
    # STEP 6 -- EMBEDDING
    # =========================================================================
    # Konversi teks yang sudah bersih ke embedding 384-dim.
    # Parameter HARUS identik dengan embedding.py:
    #   - normalize_embeddings=True -> L2 normalization -> unit vector
    #   - convert_to_numpy=True
    #   - dtype: float32 (cast eksplisit, safeguard)
    #
    # Mengapa L2 normalization wajib konsisten:
    #   SVM RBF dengan gamma='scale' dilatih pada embedding yang sudah
    #   L2-normalized. Jika saat prediksi tidak di-normalize, vektor berada
    #   di skala berbeda sehingga kernel RBF menghasilkan nilai similarity
    #   yang berbeda dan prediksi menjadi salah.
    # =========================================================================
    print_section("STEP 6: SBERT EMBEDDING")
    print("  Mengencode " + str(n_raw) + " review...")
    print("  batch_size=" + str(BATCH_SIZE) + " | normalize_embeddings=True | dtype=float32")

    t0 = time.time()
    embeddings = sbert.encode(
        df["review_clean"].tolist(),
        batch_size           = BATCH_SIZE,
        show_progress_bar    = True,
        convert_to_numpy     = True,
        normalize_embeddings = True,
    )
    # Cast ke float32 -- safeguard eksplisit, konsisten dengan embedding.py
    # SBERT sudah output float32 by default, cast ini memastikan konsistensi.
    embeddings = embeddings.astype(np.float32)

    dur_emb = time.time() - t0
    log_ok("Embedding selesai dalam " + str(round(dur_emb, 2)) + "s")
    log_ok("Shape: " + str(embeddings.shape) + " | dtype: " + str(embeddings.dtype))

    # Verifikasi L2 norm = 1.0 -- spot-check 5 sampel pertama
    norms = np.linalg.norm(embeddings[:5], axis=1)
    print("  L2 norm sample (harus ~1.0): " + str(norms.round(4).tolist()))
    if not np.allclose(norms, 1.0, atol=1e-3):
        log_warn("L2 norm tidak ~1.0 -- ada potensi inkonsistensi dengan training!")
    else:
        log_ok("L2 norm terverifikasi (~1.0) -- konsisten dengan training")

    # =========================================================================
    # STEP 7 -- LOAD BEST SVM MODEL
    # =========================================================================
    # Model yang dimuat adalah hasil akhir dari modeling.py:
    #   - Algoritma   : SVC
    #   - Kernel      : RBF
    #   - C           : 1
    #   - gamma       : scale
    #   - class_weight: balanced
    #   - probability : True (Platt Scaling aktif, predict_proba tersedia)
    #   - Dilatih pada: seluruh X_train (4794 sampel, 384-dim)
    #   - CV F1-Macro : 58.84% (Stratified 5-Fold)
    #   - Test F1-Macro: 60.06% (dari evaluasi.py)
    #
    # Model di-load sekali dan digunakan untuk semua prediksi.
    # Tidak ada fitting atau training ulang di sini.
    # =========================================================================
    print_section("STEP 7: LOAD BEST SVM MODEL")
    print("  File: " + IN_BEST_MODEL)

    t0    = time.time()
    model = joblib.load(IN_BEST_MODEL)
    log_ok("Model dimuat dalam " + str(round(time.time() - t0, 2)) + "s")
    log_ok("Tipe    : " + type(model).__name__)
    log_ok("Kernel  : " + str(model.kernel))
    log_ok("C       : " + str(model.C))
    log_ok("Gamma   : " + str(model.gamma))
    log_ok("Classes : " + str(model.classes_.tolist()) + "  (0=neg, 1=neu, 2=pos)")
    log_ok("Support vectors: " + str(model.n_support_.sum()) + " total " +
           str(model.n_support_.tolist()) + " per kelas")

    # Verifikasi model mendukung predict_proba
    has_proba = hasattr(model, "predict_proba") and model.probability
    if not has_proba:
        log_warn("Model tidak mendukung predict_proba (probability=False saat training).")
        log_warn("Kolom confidence tidak akan tersedia.")
    else:
        log_ok("predict_proba tersedia (probability=True)")

    # =========================================================================
    # STEP 8 -- PREDIKSI LABEL + CONFIDENCE SCORE
    # =========================================================================
    # model.predict() mengembalikan integer label:
    #   0 = negative | 1 = neutral | 2 = positive
    # Di-map ke string menggunakan LABEL_MAP yang dibangun dari label_classes.npy.
    #
    # model.predict_proba() mengembalikan array (N, 3):
    #   kolom 0 = prob negative
    #   kolom 1 = prob neutral
    #   kolom 2 = prob positive
    # max(axis=1) = confidence score = keyakinan model terhadap prediksinya.
    # =========================================================================
    print_section("STEP 8: PREDIKSI LABEL + CONFIDENCE SCORE")
    print("  Memprediksi " + str(n_raw) + " review...")

    t0       = time.time()
    y_pred   = model.predict(embeddings)
    dur_pred = time.time() - t0

    log_ok("Prediksi selesai dalam " + str(round(dur_pred, 3)) + "s " +
           "(" + str(int(n_raw / dur_pred)) + " sampel/detik)")

    # Map integer -> string label menggunakan LABEL_MAP dari label_classes.npy
    pred_labels = [LABEL_MAP.get(int(p), str(p)) for p in y_pred]

    # Probability (confidence) per kelas
    if has_proba:
        y_proba    = model.predict_proba(embeddings)
        confidence = y_proba.max(axis=1)      # max prob = confidence score
        conf_neg   = y_proba[:, 0]
        conf_neu   = y_proba[:, 1]
        conf_pos   = y_proba[:, 2]
        mean_conf  = float(confidence.mean())
        log_ok("Probabilitas: shape=" + str(y_proba.shape) +
               " | mean confidence=" + str(round(mean_conf, 4)))
    else:
        y_proba    = None
        confidence = [None] * n_raw
        conf_neg   = [None] * n_raw
        conf_neu   = [None] * n_raw
        conf_pos   = [None] * n_raw

    # =========================================================================
    # STEP 9 -- SUSUN DATAFRAME OUTPUT
    # =========================================================================
    print_section("STEP 9: SUSUN DATAFRAME HASIL")

    df_out = pd.DataFrame({
        "review_original" : df["review_original"].values,
        "review_clean"    : df["review_clean"].values,
        "word_count"      : df["word_count"].values,
        "prediction"      : pred_labels,
        "confidence"      : [round(float(c), 4) if c is not None else None
                             for c in confidence],
        "conf_negative"   : [round(float(c), 4) if c is not None else None
                             for c in conf_neg],
        "conf_neutral"    : [round(float(c), 4) if c is not None else None
                             for c in conf_neu],
        "conf_positive"   : [round(float(c), 4) if c is not None else None
                             for c in conf_pos],
    })

    # Flag review dengan confidence rendah
    # Threshold default 0.50 -- bisa disesuaikan sesuai kebutuhan produksi
    df_out["low_confidence"] = df_out["confidence"].apply(
        lambda c: True if (c is not None and c < CONF_THRESHOLD) else False
    )

    # Flag review yang terlalu pendek
    df_out["short_review"] = df["word_count"] < MIN_WORDS

    # Tambahkan timestamp prediksi
    df_out["prediction_time"] = timestamp

    # Tambahkan kolom ID di posisi pertama (lebih profesional untuk output)
    df_out.insert(0, "id", range(1, len(df_out) + 1))

    log_ok("DataFrame output: " + str(df_out.shape[0]) + " baris x " +
           str(df_out.shape[1]) + " kolom")
    log_ok("Kolom: " + str(list(df_out.columns)))

    # =========================================================================
    # STEP 10 -- SIMPAN OUTPUT
    # =========================================================================
    print_section("STEP 10: SIMPAN HASIL KE CSV")

    df_out.to_csv(OUT_PREDICT, index=False, encoding="utf-8-sig")
    size_kb = os.path.getsize(OUT_PREDICT) // 1024
    log_ok("Tersimpan: " + OUT_PREDICT + " (" + str(size_kb) + " KB)")
    log_ok("Total prediksi: " + str(len(df_out)) + " review")

    # =========================================================================
    # STEP 11 -- AUDIT & RINGKASAN HASIL
    # =========================================================================
    print_section("STEP 11: AUDIT & RINGKASAN HASIL PREDIKSI")

    # -- Distribusi label prediksi ------------------------------------------
    print("\n  Distribusi Label Prediksi:")
    dist = df_out["prediction"].value_counts()
    for label in ["positive", "negative", "neutral"]:
        count = dist.get(label, 0)
        pct   = count / len(df_out) * 100
        bar   = "#" * int(pct / 5)
        print("    " + label.ljust(12) + ": " +
              str(count).rjust(4) + " (" + str(round(pct, 1)).rjust(5) + "%)  " + bar)

    # -- Rata-rata confidence per kelas prediksi ----------------------------
    if has_proba:
        print("\n  Rata-rata Confidence per Kelas Prediksi:")
        for label in ["negative", "neutral", "positive"]:
            mask = df_out["prediction"] == label
            if mask.sum() > 0:
                mean_c = df_out.loc[mask, "confidence"].mean()
                min_c  = df_out.loc[mask, "confidence"].min()
                max_c  = df_out.loc[mask, "confidence"].max()
                print("    " + label.ljust(12) +
                      ": mean=" + str(round(mean_c, 4)) +
                      " | min=" + str(round(min_c, 4)) +
                      " | max=" + str(round(max_c, 4)))

    # -- Low confidence summary --------------------------------------------
    n_low = df_out["low_confidence"].sum()
    if n_low > 0:
        log_warn(str(n_low) + "/" + str(len(df_out)) +
                 " review confidence < " + str(CONF_THRESHOLD) +
                 " -> disarankan human review.")
        lc_df = df_out[df_out["low_confidence"]].head(5)
        print("\n  Review Low-Confidence (maks 5):")
        for _, row in lc_df.iterrows():
            enc  = sys.stdout.encoding or "utf-8"
            line = ("    [" + str(row["prediction"]).ljust(8) +
                    " | conf=" + str(round(row["confidence"], 3)) + "] " +
                    str(row["review_clean"])[:55])
            print(line.encode(enc, errors="replace").decode(enc))
    else:
        log_ok("Semua prediksi confidence >= " + str(CONF_THRESHOLD))

    # -- Short review summary -----------------------------------------------
    n_short_out = df_out["short_review"].sum()
    if n_short_out > 0:
        log_warn(str(n_short_out) + " review sangat pendek (< " + str(MIN_WORDS) +
                 " kata) -- akurasi prediksinya perlu diwaspadai.")

    # -- Preview tabel hasil (semua baris) ----------------------------------
    print("\n  Preview Hasil (semua baris):")
    print("  " + "No".ljust(4) + " " +
          "Review (cleaned, maks 40 kar)".ljust(42) + " " +
          "Label".ljust(12) + " " +
          "Conf".rjust(6) + " " +
          "Low?".rjust(5))
    print("  " + "-" * 72)

    enc = sys.stdout.encoding or "utf-8"
    for _, row in df_out.iterrows():
        review_short = str(row["review_clean"])[:40]
        if len(str(row["review_clean"])) > 40:
            review_short = review_short[:-2] + ".."
        conf_str = (str(round(row["confidence"], 4)) if row["confidence"] is not None
                    else " N/A ")
        low_flag = "  [!]" if row["low_confidence"] else "     "
        line = ("  " + str(row["id"]).ljust(4) + " " +
                review_short.ljust(42) + " " +
                str(row["prediction"]).ljust(12) + " " +
                conf_str.rjust(6) + low_flag)
        print(line.encode(enc, errors="replace").decode(enc))

    # =========================================================================
    # PENUTUP
    # =========================================================================
    elapsed = time.time() - t_start
    th, rem = divmod(int(elapsed), 3600)
    tm, ts  = divmod(rem, 60)

    closing = (
        "\n" + "=" * 65 + "\n" +
        "  PREDIKSI SELESAI\n" +
        "\n  Model    : SVM RBF | C=1 | gamma=scale\n" +
        "  SBERT    : " + SBERT_MODEL + "\n" +
        "  Labels   : data/label_classes.npy (bukan hardcode)\n" +
        "  Input    : " + IN_SAMPLE_CSV + "\n" +
        "  Output   : " + OUT_PREDICT + "\n" +
        "\n  STATISTIK:\n" +
        "    Total diprediksi : " + str(len(df_out)) + "\n" +
        "    Low confidence   : " + str(n_low) + " review (< " + str(CONF_THRESHOLD) + ")\n" +
        "    Review pendek    : " + str(n_short_out) + " review (< " + str(MIN_WORDS) + " kata)\n" +
        "    Waktu embedding  : " + str(round(dur_emb, 2)) + "s\n" +
        "    Waktu prediksi   : " + str(round(dur_pred, 3)) + "s\n" +
        "    Waktu total      : " + str(th) + "h " + str(tm) + "m " + str(ts).zfill(2) + "s\n" +
        "\n  KOLOM OUTPUT (sample_predictions.csv):\n" +
        "    id               : nomor urut (1, 2, 3, ...)\n" +
        "    review_original  : teks review asli\n" +
        "    review_clean     : teks setelah preprocessing\n" +
        "    word_count       : jumlah kata setelah cleaning\n" +
        "    prediction       : negative / neutral / positive\n" +
        "    confidence       : max probability (keyakinan model)\n" +
        "    conf_negative    : prob kelas negative\n" +
        "    conf_neutral     : prob kelas neutral\n" +
        "    conf_positive    : prob kelas positive\n" +
        "    low_confidence   : True jika confidence < " + str(CONF_THRESHOLD) + "\n" +
        "    short_review     : True jika < " + str(MIN_WORDS) + " kata\n" +
        "    prediction_time  : timestamp saat prediksi dijalankan\n" +
        "\n  CATATAN TEKNIS:\n" +
        "    - Label mapping dari label_classes.npy (bukan hardcode)\n" +
        "    - Preprocessing identik dengan preprocessing.py\n" +
        "    - SBERT L2-normalized, identik dengan embedding.py\n" +
        "    - Model dari output/svm_sbert_best_model.joblib\n" +
        "    - Tidak ada training ulang di script ini\n" +
        "=" * 65
    )
    print(C_HEAD + closing + C_RST)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
