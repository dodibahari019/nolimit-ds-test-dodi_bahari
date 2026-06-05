"""
=============================================================================
TAHAP EVALUASI FINAL: SVM + SBERT Embedding
=============================================================================
Proyek  : Sentiment Analysis Review (Indonesian)
Tahap   : Evaluasi Final - Generalization Performance pada Test Set
Versi   : v1.0

=============================================================================
RUANG LINGKUP TAHAP INI
=============================================================================
  Tahap ini adalah EVALUASI FINAL - satu-satunya kali X_test disentuh.
  Model yang dievaluasi: svm_sbert_best_model.joblib (dari tahap Modeling)

  YANG DILAKUKAN:
    [A] Load model terbaik + X_test + y_test
    [B] Prediksi & hitung semua metrik evaluasi
    [C] Deteksi generalization gap (CV vs Test)
    [D] 9 visualisasi komprehensif
    [E] Analisis error per kelas & rekomendasi langkah selanjutnya
    [F] Simpan log lengkap

  PRINSIP:
    !! X_test hanya disentuh SEKALI di sini - tidak untuk tuning !!
    !! Hasil ini adalah estimasi performa di dunia nyata !!

=============================================================================
INPUT:
  output/svm_sbert_best_model.joblib     -> Model terbaik dari tahap Modeling
  data/X_test_emb.npy                    -> (N_test x 384) float32
  data/y_test.npy                        -> Label test (int)
  data/label_classes.npy                 -> ['negative', 'neutral', 'positive']

  OPSIONAL (untuk perbandingan CV):
  output/svm_sbert_baseline_model.joblib -> Baseline (untuk prediksi baseline)
  output/svm_sbert_tuning_results.csv    -> CV scores dari tahap Modeling

OUTPUT:
  output/eval_metrics_summary.csv        -> Semua metrik dalam satu tabel
  output/eval_classification_report.txt  -> Classification report lengkap
  output/viz_eval_confusion_matrix.png   -> CM test set (FINAL)
  output/viz_eval_per_class_metrics.png  -> F1/P/R per kelas (test set)
  output/viz_eval_cv_vs_test_gap.png     -> Gap CV vs Test (generalization)
  output/viz_eval_roc_multiclass.png     -> ROC curve per kelas (OvR)
  output/viz_eval_pr_curve.png           -> Precision-Recall curve per kelas
  output/viz_eval_confidence_dist.png    -> Distribusi confidence score
  output/viz_eval_error_analysis.png     -> Analisis error per pasangan kelas
  output/viz_eval_calibration.png        -> Kalibrasi probabilitas (reliability diagram)
  output/viz_eval_summary_dashboard.png  -> Dashboard ringkasan UNGGULAN
  output/evaluation_final_log.txt        -> Log lengkap + rekomendasi

=============================================================================
INSTALL:
  pip install scikit-learn numpy pandas joblib matplotlib colorama
=============================================================================
"""

import os
import sys
import io
# Fix Windows encoding issues
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
    brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(CURRENT_DIR, "data")
OUTPUT_DIR  = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input wajib
IN_BEST_MODEL = os.path.join(OUTPUT_DIR, "svm_sbert_best_model.joblib")
IN_X_TEST     = os.path.join(DATA_DIR,   "X_test_emb.npy")
IN_Y_TEST     = os.path.join(DATA_DIR,   "y_test.npy")
IN_CLASSES    = os.path.join(DATA_DIR,   "label_classes.npy")

# Input opsional (untuk perbandingan)
IN_BASELINE   = os.path.join(OUTPUT_DIR, "svm_sbert_baseline_model.joblib")
IN_CV_RESULTS = os.path.join(OUTPUT_DIR, "svm_sbert_tuning_results.csv")

# CV scores dari tahap modeling (update sesuai output modeling kamu)
CV_BEST_F1_MACRO    = 0.5884   # Dari log modeling - CV F1-Macro best model
CV_BEST_F1_WEIGHTED = 0.8205   # CV F1-Weighted best model
CV_BEST_ACCURACY    = 0.8062   # CV Accuracy best model
CV_BASELINE_F1      = 0.5707   # CV F1-Macro baseline

# Nama best model params (untuk label viz)
BEST_KERNEL = "RBF"
BEST_C      = 1
BEST_GAMMA  = "scale"

# =============================================================================
# WARNA & STYLE
# =============================================================================
VIZ_BG       = "#f8f9fa"
VIZ_GRID_CLR = "#dee2e6"
VIZ_NEG      = "#e74c3c"
VIZ_NEU      = "#f39c12"
VIZ_POS      = "#27ae60"
VIZ_BEST     = "#2980b9"
VIZ_BASELINE = "#95a5a6"
VIZ_CV       = "#8e44ad"
VIZ_TEST     = "#e67e22"

CLASS_COLORS = [VIZ_NEG, VIZ_NEU, VIZ_POS]

# =============================================================================
# LOGGING
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

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(str(msg))

def log_ok(msg):
    print(C_OK + f"  [OK]   {msg}" + C_RST)
    log_lines.append(f"  [OK]   {msg}")

def log_warn(msg):
    print(C_WARN + f"  [WARN] {msg}" + C_RST)
    log_lines.append(f"  [WARN] {msg}")

def log_err(msg):
    print(C_ERR + f"  [ERR]  {msg}" + C_RST)
    log_lines.append(f"  [ERR]  {msg}")

def print_section(title):
    line = "=" * 65
    log(f"\n{line}")
    log(f"  {title}")
    log(line)


# =============================================================================
# VIZ 1 - CONFUSION MATRIX FINAL (Test Set)
# =============================================================================
def viz_confusion_matrix_test(cm, classes, metrics, output_dir):
    """
    CM final pada test set. Panel kiri: raw count. Panel kanan: normalized.
    Ini adalah evaluasi sebenarnya - bukan CV lagi.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    n = len(classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    for ax, data, fmt, title_suffix in [
        (ax1, cm,      "d",     "Raw Count"),
        (ax2, cm_norm, ".2f",   "Normalized (per baris Aktual)"),
    ]:
        ax.set_facecolor(VIZ_BG)
        im = ax.imshow(data, cmap="Blues",
                       vmin=0, vmax=(1 if fmt == ".2f" else cm.max()))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(n):
            for j in range(n):
                val = data[i, j]
                thr = (0.55 if fmt == ".2f" else cm.max() * 0.55)
                txt_color = "white" if val > thr else "black"
                label = f"{val:{fmt}}" if fmt == "d" else f"{val*100:.1f}%"
                if fmt == "d":
                    label += f"\n({cm_norm[i,j]*100:.1f}%)"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=txt_color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
        ax.set_yticklabels([c.capitalize() for c in classes], fontsize=10)
        ax.set_xlabel("Prediksi", fontsize=11, labelpad=6)
        ax.set_ylabel("Aktual", fontsize=11, labelpad=6)
        ax.set_title(
            f"Confusion Matrix - {title_suffix}\n"
            f"Test Set Final | F1-Macro: {metrics['f1_macro']*100:.2f}%",
            fontsize=10, fontweight="bold", color="#212529", pad=10)

    fig.suptitle(
        f"Confusion Matrix FINAL - SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA}\n"
        f"Test Set: {cm.sum()} sampel | 3 Kelas Sentimen Bahasa Indonesia\n"
        f"F1-Macro={metrics['f1_macro']*100:.2f}%  "
        f"Accuracy={metrics['accuracy']*100:.2f}%  "
        f"F1-Weighted={metrics['f1_weighted']*100:.2f}%",
        fontsize=12, fontweight="bold", color="#212529", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_eval_confusion_matrix.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_confusion_matrix.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 2 - PER CLASS METRICS (Test Set)
# =============================================================================
def viz_per_class_metrics(y_test, y_pred, classes, output_dir):
    """
    F1, Precision, Recall per kelas pada test set.
    Kotak merah di kelas dengan F1 terendah - biasanya Neutral.
    """
    n = len(classes)
    f1s  = f1_score(y_test, y_pred, average=None, zero_division=0)
    prcs = precision_score(y_test, y_pred, average=None, zero_division=0)
    recs = recall_score(y_test, y_pred, average=None, zero_division=0)

    # Support (jumlah sampel per kelas di test set)
    supports = [(y_test == i).sum() for i in range(n)]

    metrics_names = ["F1-Score", "Precision", "Recall"]
    x = np.arange(len(metrics_names))
    w = 0.25

    fig, (ax_bar, ax_radar) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    # -- Bar chart kiri ----------------------------------------------------
    ax_bar.set_facecolor(VIZ_BG)
    worst_cls = int(np.argmin(f1s))

    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        offset = (i - 1) * w
        vals   = [f1s[i], prcs[i], recs[i]]
        bars   = ax_bar.bar(x + offset, vals, w,
                            label=f"{cls.capitalize()} (n={supports[i]})",
                            color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.012,
                        f"{val*100:.1f}%",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color="#212529")

        # Tandai kelas terburuk
        if i == worst_cls:
            for bar in bars:
                bar.set_edgecolor("#c0392b")
                bar.set_linewidth(2.0)
                ax_bar.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.035,
                            "[!]", ha="center", va="bottom",
                            fontsize=10, color="#c0392b")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics_names, fontsize=11)
    ax_bar.set_ylabel("Skor (Test Set)", fontsize=11, labelpad=8)
    ax_bar.set_ylim(0, 1.20)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_bar.set_title(
        "F1 / Precision / Recall per Kelas - Test Set FINAL\n"
        f"Terbaik: {classes[int(np.argmax(f1s))].capitalize()} "
        f"({f1s[int(np.argmax(f1s))]*100:.1f}%)  |  "
        f"Terburuk: {classes[worst_cls].capitalize()} "
        f"({f1s[worst_cls]*100:.1f}%)\n"
        "[!] = kelas dengan F1 terendah",
        fontsize=10, fontweight="bold", color="#212529", pad=10)
    ax_bar.legend(fontsize=9, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax_bar.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # -- Radar chart kanan -------------------------------------------------
    ax_radar.set_facecolor(VIZ_BG)
    ax_radar.remove()
    ax_radar = fig.add_subplot(1, 2, 2, projection="polar")

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]

    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        vals_radar = [f1s[i], prcs[i], recs[i]]
        vals_radar += vals_radar[:1]
        ax_radar.plot(angles, vals_radar, color=color,
                      linewidth=2.0, marker="o", markersize=6,
                      label=cls.capitalize())
        ax_radar.fill(angles, vals_radar, color=color, alpha=0.08)

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics_names, fontsize=10)
    ax_radar.set_ylim(0, 1)
    ax_radar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.grid(color=VIZ_GRID_CLR, linewidth=0.8)
    ax_radar.set_title(
        "Radar: F1/P/R per Kelas\n(luas area = performa keseluruhan)",
        fontsize=10, fontweight="bold", color="#212529", pad=20)
    ax_radar.legend(fontsize=9, loc="upper right",
                    bbox_to_anchor=(1.35, 1.1), framealpha=0.85)

    fig.suptitle(
        f"Metrik per Kelas Sentimen - SVM {BEST_KERNEL} C={BEST_C} | Test Set Final\n"
        f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0)*100:.2f}%  |  "
        f"Weighted F1: {f1_score(y_test, y_pred, average='weighted', zero_division=0)*100:.2f}%",
        fontsize=12, fontweight="bold", color="#212529", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_eval_per_class_metrics.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_per_class_metrics.png ({os.path.getsize(path)//1024} KB)")
    return path, f1s, prcs, recs


# =============================================================================
# VIZ 3 - CV vs TEST GAP (Generalization Analysis)
# =============================================================================
def viz_cv_vs_test_gap(metrics_test, output_dir):
    """
    Perbandingan CV score (dari tahap Modeling) vs Test score (sekarang).
    Menunjukkan apakah model overfit pada CV atau generalize dengan baik.
    GAP > 5% = perlu diinvestigasi.
    """
    metrics_display = [
        ("F1-Macro\n(Metrik Utama)", CV_BEST_F1_MACRO,    metrics_test["f1_macro"]),
        ("F1-Weighted",             CV_BEST_F1_WEIGHTED,  metrics_test["f1_weighted"]),
        ("Accuracy",                CV_BEST_ACCURACY,     metrics_test["accuracy"]),
    ]
    labels = [m[0] for m in metrics_display]
    cv_vals   = [m[1] for m in metrics_display]
    test_vals = [m[2] for m in metrics_display]
    gaps      = [t - c for c, t in zip(cv_vals, test_vals)]

    x = np.arange(len(labels))
    w = 0.30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    # -- Panel kiri: grouped bar --------------------------------------------
    ax1.set_facecolor(VIZ_BG)
    b_cv   = ax1.bar(x - w/2, cv_vals, w, label="CV Train Score",
                     color=VIZ_CV, alpha=0.80, edgecolor="white")
    b_test = ax1.bar(x + w/2, test_vals, w, label="Test Score (Final)",
                     color=VIZ_TEST, alpha=0.85, edgecolor="white")

    for bar, val in zip(list(b_cv) + list(b_test),
                        cv_vals + test_vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.008,
                 f"{val*100:.2f}%",
                 ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#212529")

    for i, gap in enumerate(gaps):
        color_g = "#27ae60" if gap >= 0 else "#e74c3c"
        mid_x   = x[i]
        y_top   = max(cv_vals[i], test_vals[i]) + 0.04
        ax1.annotate(
            f"{gap*100:+.2f}%",
            xy=(mid_x, y_top),
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=color_g,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=color_g,
                      linewidth=1.5, alpha=0.9))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Skor", fontsize=11, labelpad=8)
    ax1.set_ylim(0, 1.18)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax1.set_title(
        "CV Train Score vs Test Score\n"
        "Angka di atas = gap (+ artinya Test > CV = generalize baik)\n"
        "Gap besar & negatif = potensi overfitting",
        fontsize=10, fontweight="bold", color="#212529", pad=10)
    ax1.legend(fontsize=10, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    # -- Panel kanan: interpretasi gap ------------------------------------
    ax2.set_facecolor(VIZ_BG)
    ax2.axis("off")

    main_gap = metrics_test["f1_macro"] - CV_BEST_F1_MACRO
    gap_status = "[OK] GENERALIZE BAIK" if abs(main_gap) <= 0.05 else \
                 ("[WARN] TEST > CV (Unusual, cek data)" if main_gap > 0.05 else
                  "[ERR] OVERFIT (Gap besar)")

    interpretation = [
        ("HASIL EVALUASI TEST SET FINAL", "", "#212529", 14, True),
        ("", "", "", 10, False),
        (f"F1-Macro (Test):", f"{metrics_test['f1_macro']*100:.4f}%",
         VIZ_TEST, 12, True),
        (f"F1-Macro (CV):",   f"{CV_BEST_F1_MACRO*100:.4f}%",
         VIZ_CV,  12, False),
        (f"Generalization Gap:", f"{main_gap*100:+.4f}%",
         "#27ae60" if main_gap >= -0.03 else "#e74c3c", 12, True),
        ("", "", "", 8, False),
        ("Status Generalisasi:", gap_status,
         "#27ae60" if abs(main_gap) <= 0.05 else "#e74c3c", 12, True),
        ("", "", "", 8, False),
        ("INTERPRETASI:", "", "#212529", 11, True),
    ]

    # Interpretasi otomatis
    if abs(main_gap) <= 0.05:
        interpretation += [
            ("[OK] Model tidak overfit.", "",  "#27ae60", 10, False),
            ("[OK] CV score cukup representatif.", "", "#27ae60", 10, False),
            ("Langkah selanjutnya:", "", "#212529", 10, True),
            ("-> Lihat F1 per kelas (Neutral?)", "", "#2980b9", 10, False),
            ("-> Pertimbangkan SMOTE jika Neutral rendah", "",
             "#2980b9", 10, False),
            ("-> Atau coba IndoBERT/IndoBERTweet", "", "#2980b9", 10, False),
        ]
    elif main_gap > 0.05:
        interpretation += [
            ("[WARN] Test lebih tinggi dari CV - tidak umum.", "",
             "#f39c12", 10, False),
            ("Kemungkinan: distribusi test set lebih mudah.", "",
             "#f39c12", 10, False),
            ("-> Periksa distribusi kelas di test set.", "",
             "#e74c3c", 10, False),
        ]
    else:
        interpretation += [
            ("[ERR] Model overfit - test jauh di bawah CV.", "",
             "#e74c3c", 10, False),
            ("Kemungkinan: data leakage atau train/test mismatch.", "",
             "#e74c3c", 10, False),
            ("-> Periksa feature extraction pipeline.", "",
             "#e74c3c", 10, False),
            ("-> Periksa apakah test/train di-split sebelum SBERT.", "",
             "#e74c3c", 10, False),
        ]

    y_pos = 0.96
    for left, right, color, size, bold in interpretation:
        if not left and not right:
            y_pos -= 0.02
            continue
        text = f"{left}  {right}" if right else left
        ax2.text(0.05, y_pos, text,
                 transform=ax2.transAxes,
                 fontsize=size, color=color,
                 fontweight="bold" if bold else "normal",
                 va="top")
        y_pos -= (size / 120)

    fig.suptitle(
        "Analisis Generalisasi: CV Score vs Test Score Final\n"
        f"Model: SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA} | "
        f"Test Set Evaluasi (BUKAN CV)",
        fontsize=12, fontweight="bold", color="#212529", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_eval_cv_vs_test_gap.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_cv_vs_test_gap.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 4 - ROC CURVE MULTICLASS (OvR)
# =============================================================================
def viz_roc_multiclass(y_test, y_proba, classes, output_dir):
    """
    ROC curve per kelas menggunakan One-vs-Rest approach.
    AUC per kelas + macro-average AUC.
    Hanya bisa dibuat jika model punya probability=True.
    """
    n = len(classes)
    y_bin = label_binarize(y_test, classes=list(range(n)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    auc_scores = []
    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{cls.capitalize()} (AUC = {roc_auc:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.04, color=color)

    # Macro-average
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="#2c3e50", linewidth=2.5,
            linestyle="--", label=f"Macro-Avg (AUC = {macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, alpha=0.5,
            label="Random Classifier (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, labelpad=8)
    ax.set_title(
        f"ROC Curve per Kelas - One-vs-Rest (OvR)\n"
        f"SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA} | Test Set Final\n"
        f"Macro AUC = {macro_auc:.4f}  |  "
        f"AUC > 0.7 = cukup baik untuk 3-class imbalanced",
        fontsize=10, fontweight="bold", color="#212529", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax.grid(color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "viz_eval_roc_multiclass.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_roc_multiclass.png ({os.path.getsize(path)//1024} KB)")
    return path, macro_auc, auc_scores


# =============================================================================
# VIZ 5 - PRECISION-RECALL CURVE
# =============================================================================
def viz_pr_curve(y_test, y_proba, classes, output_dir):
    """
    PR curve per kelas. Lebih informatif dari ROC untuk imbalanced dataset.
    Average Precision (AP) = area di bawah PR curve.
    """
    n = len(classes)
    y_bin = label_binarize(y_test, classes=list(range(n)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    ap_scores = []
    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap            = average_precision_score(y_bin[:, i], y_proba[:, i])
        ap_scores.append(ap)
        baseline_p = y_bin[:, i].mean()
        ax.plot(rec, prec, color=color, linewidth=2.2,
                label=f"{cls.capitalize()} (AP = {ap:.4f}, "
                      f"baseline = {baseline_p:.3f})")
        ax.fill_between(rec, prec, alpha=0.04, color=color)

        # Garis baseline per kelas
        ax.axhline(baseline_p, color=color, linewidth=0.8,
                   linestyle=":", alpha=0.6)

    mean_ap = np.mean(ap_scores)
    ax.set_xlabel("Recall", fontsize=11, labelpad=8)
    ax.set_ylabel("Precision", fontsize=11, labelpad=8)
    ax.set_title(
        f"Precision-Recall Curve per Kelas - OvR\n"
        f"SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA} | Test Set Final\n"
        f"Mean AP = {mean_ap:.4f}  |  "
        f"Titik putus-putus = baseline (prevalensi kelas)",
        fontsize=10, fontweight="bold", color="#212529", pad=10)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax.grid(color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "viz_eval_pr_curve.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_pr_curve.png ({os.path.getsize(path)//1024} KB)")
    return path, mean_ap, ap_scores


# =============================================================================
# VIZ 6 - DISTRIBUSI CONFIDENCE SCORE
# =============================================================================
def viz_confidence_dist(y_test, y_pred, y_proba, classes, output_dir):
    """
    Distribusi confidence (max probability) per kelas.
    Memisahkan prediksi benar vs salah.
    Confidence rendah = model tidak yakin -> bisa di-flag untuk human review.
    """
    n = len(classes)
    max_proba = y_proba.max(axis=1)
    correct   = (y_pred == y_test)

    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    fig.patch.set_facecolor(VIZ_BG)

    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        ax = axes[i]
        ax.set_facecolor(VIZ_BG)

        mask_cls = (y_pred == i)
        conf_correct = max_proba[mask_cls & correct]
        conf_wrong   = max_proba[mask_cls & ~correct]

        bins = np.linspace(0, 1, 25)
        if len(conf_correct) > 0:
            ax.hist(conf_correct, bins=bins, color=color,
                    alpha=0.70, label=f"Benar (n={len(conf_correct)})",
                    edgecolor="white")
        if len(conf_wrong) > 0:
            ax.hist(conf_wrong, bins=bins, color="#c0392b",
                    alpha=0.55, label=f"Salah (n={len(conf_wrong)})",
                    edgecolor="white")

        total_cls = mask_cls.sum()
        acc_cls   = conf_correct.shape[0] / total_cls if total_cls > 0 else 0
        mean_c    = max_proba[mask_cls].mean() if total_cls > 0 else 0

        ax.axvline(0.5, color="#2c3e50", linestyle="--",
                   linewidth=1.5, alpha=0.7, label="Threshold 0.5")
        ax.set_xlabel("Confidence Score (max prob)", fontsize=10, labelpad=6)
        ax.set_ylabel("Jumlah Sampel", fontsize=10, labelpad=6)
        ax.set_title(
            f"Prediksi: {cls.capitalize()}\n"
            f"Total diprediksi: {total_cls} | Akurasi: {acc_cls*100:.1f}%\n"
            f"Mean confidence: {mean_c:.3f}",
            fontsize=10, fontweight="bold", color=color, pad=8)
        ax.legend(fontsize=8, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
        ax.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 1)

    overall_conf_mean = max_proba.mean()
    fig.suptitle(
        f"Distribusi Confidence Score per Kelas Prediksi - Test Set Final\n"
        f"Mean confidence keseluruhan: {overall_conf_mean:.4f}  |  "
        f"Merah = prediksi salah | Biru/Warna = prediksi benar\n"
        f"Confidence rendah (<0.5) biasanya prediksi kelas minoritas",
        fontsize=12, fontweight="bold", color="#212529", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_eval_confidence_dist.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_confidence_dist.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 7 - ANALISIS ERROR (Misprediction Patterns)
# =============================================================================
def viz_error_analysis(cm, classes, output_dir):
    """
    Analisis error: mana kelas yang paling sering salah diprediksi jadi apa.
    Menggunakan stacked bar dari confusion matrix (tanpa diagonal = benar).
    """
    n = len(classes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    # Panel kiri: error rate per kelas aktual
    ax1.set_facecolor(VIZ_BG)
    error_rates  = [(cm[i].sum() - cm[i, i]) / cm[i].sum()
                    for i in range(n)]
    recall_rates = [cm[i, i] / cm[i].sum() for i in range(n)]

    x = np.arange(n)
    w = 0.35
    b1 = ax1.bar(x - w/2, recall_rates, w,
                 color=[c for c in CLASS_COLORS], alpha=0.80,
                 edgecolor="white", label="Recall (Benar Diprediksi)")
    b2 = ax1.bar(x + w/2, error_rates, w,
                 color="#c0392b", alpha=0.60,
                 edgecolor="white", label="Error Rate (Salah Diprediksi)")

    for bar, val in zip(list(b1) + list(b2),
                        recall_rates + error_rates):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.012,
                 f"{val*100:.1f}%",
                 ha="center", va="bottom", fontsize=9.5,
                 fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
    ax1.set_ylabel("Proporsi", fontsize=11, labelpad=8)
    ax1.set_ylim(0, 1.20)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax1.set_title(
        "Recall vs Error Rate per Kelas Aktual\n"
        "Error Rate = proporsi salah diklasifikasi dari total aktual\n"
        "Kelas dengan Error Rate tinggi = bottleneck model",
        fontsize=10, fontweight="bold", color="#212529", pad=10)
    ax1.legend(fontsize=9, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel kanan: error ke mana (off-diagonal heatmap)
    ax2.set_facecolor(VIZ_BG)
    error_matrix = cm.copy().astype(float)
    np.fill_diagonal(error_matrix, 0)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    error_rate_matrix = error_matrix / row_sums

    im = ax2.imshow(error_rate_matrix, cmap="Reds",
                    vmin=0, vmax=error_rate_matrix.max() * 1.1 or 0.1)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04,
                 label="Error Rate (dari aktual)")

    for i in range(n):
        for j in range(n):
            if i != j:
                val    = error_rate_matrix[i, j]
                count  = int(cm[i, j])
                txt_c  = "white" if val > error_rate_matrix.max() * 0.6 else "black"
                ax2.text(j, i, f"{val*100:.1f}%\n(n={count})",
                         ha="center", va="center",
                         fontsize=10, fontweight="bold", color=txt_c)
            else:
                ax2.text(j, i, "-", ha="center", va="center",
                         fontsize=14, color="#aaaaaa")

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([f"Pred: {c.capitalize()}" for c in classes],
                        fontsize=9.5)
    ax2.set_yticklabels([f"Aktual: {c.capitalize()}" for c in classes],
                        fontsize=9.5)
    ax2.set_title(
        "Pola Error: Kelas Aktual -> Diprediksi Salah Jadi Apa\n"
        "(diagonal = benar, tidak ditampilkan)\n"
        "Warna merah gelap = sering salah ke sana",
        fontsize=10, fontweight="bold", color="#212529", pad=10)

    fig.suptitle(
        f"Analisis Error / Misprediction Pattern - Test Set Final\n"
        f"SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA}",
        fontsize=12, fontweight="bold", color="#212529", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_eval_error_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_error_analysis.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 8 - KALIBRASI PROBABILITAS (Reliability Diagram)
# =============================================================================
def viz_calibration(y_test, y_proba, classes, output_dir):
    """
    Reliability diagram: apakah confidence score 70% benar-benar 70% akurat?
    Model yang well-calibrated = garis diagonal.
    Bermanfaat untuk threshold tuning & confidence-based filtering.
    """
    n = len(classes)
    y_bin = label_binarize(y_test, classes=list(range(n)))

    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    fig.patch.set_facecolor(VIZ_BG)

    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        ax = axes[i]
        ax.set_facecolor(VIZ_BG)

        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin[:, i], y_proba[:, i], n_bins=8, strategy="uniform")
            brier  = brier_score_loss(y_bin[:, i], y_proba[:, i])

            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
                    alpha=0.6, label="Perfect Calibration")
            ax.plot(mean_pred, frac_pos, color=color,
                    marker="o", linewidth=2.2, markersize=8,
                    label=f"{cls.capitalize()} (Brier={brier:.4f})")
            ax.fill_between(mean_pred, frac_pos, mean_pred,
                            alpha=0.08, color=color,
                            label="Calibration Gap")
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data\nfor calibration",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color=color)

        ax.set_xlabel("Mean Predicted Probability", fontsize=10, labelpad=6)
        ax.set_ylabel("Fraction of Positives (Actual)", fontsize=10, labelpad=6)
        ax.set_title(
            f"Kalibrasi: {cls.capitalize()}\n"
            f"Di atas diagonal = under-confident\n"
            f"Di bawah diagonal = over-confident",
            fontsize=10, fontweight="bold", color=color, pad=8)
        ax.legend(fontsize=8.5, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
        ax.grid(color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        "Reliability Diagram (Kalibrasi Probabilitas) - Test Set Final\n"
        "Garis putus-putus = kalibrasi sempurna | "
        "Brier Score rendah = kalibrasi baik (maks 1.0)\n"
        "SVM dengan Platt Scaling cenderung over-confident untuk kelas minoritas",
        fontsize=12, fontweight="bold", color="#212529", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_eval_calibration.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_calibration.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 9 - SUMMARY DASHBOARD (UNGGULAN)
# =============================================================================
def viz_summary_dashboard(metrics, cm, classes, f1s, prcs, recs,
                           macro_auc, ap_scores, output_dir):
    """
    Dashboard ringkasan satu halaman:
    - Scorecard metrik utama
    - Mini CM
    - F1 per kelas
    - Rekomendasi langkah selanjutnya
    """
    n = len(classes)
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(VIZ_BG)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.40)

    # -- SCORECARD ATAS ----------------------------------------------------
    score_data = [
        ("F1-Macro\n(Metrik Utama)", metrics["f1_macro"],
         CV_BEST_F1_MACRO, VIZ_BEST),
        ("F1-Weighted",             metrics["f1_weighted"],
         CV_BEST_F1_WEIGHTED, "#16a085"),
        ("Accuracy",               metrics["accuracy"],
         CV_BEST_ACCURACY, "#8e44ad"),
        ("Macro AUC-ROC",          macro_auc,  None, "#d35400"),
    ]

    for col_idx, (metric_name, test_val, cv_val, color) in enumerate(score_data):
        ax_s = fig.add_subplot(gs[0, col_idx])
        ax_s.set_facecolor("white")
        for sp in ax_s.spines.values():
            sp.set_visible(True)
            sp.set_color(color)
            sp.set_linewidth(2.5)

        ax_s.text(0.5, 0.72, f"{test_val*100:.2f}%",
                  transform=ax_s.transAxes,
                  fontsize=22, fontweight="bold",
                  ha="center", va="center", color=color)
        ax_s.text(0.5, 0.42, metric_name,
                  transform=ax_s.transAxes,
                  fontsize=10, ha="center", va="center",
                  color="#212529", fontweight="bold")
        if cv_val is not None:
            gap = test_val - cv_val
            gap_c = "#27ae60" if gap >= -0.02 else "#e74c3c"
            ax_s.text(0.5, 0.15, f"CV: {cv_val*100:.2f}%  Gap: {gap*100:+.2f}%",
                      transform=ax_s.transAxes,
                      fontsize=8.5, ha="center", va="center",
                      color=gap_c)
        ax_s.axis("off")

    # -- MINI CONFUSION MATRIX ---------------------------------------------
    ax_cm = fig.add_subplot(gs[1, 0:2])
    ax_cm.set_facecolor(VIZ_BG)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            txt_c = "white" if cm_norm[i, j] > 0.55 else "black"
            ax_cm.text(j, i,
                       f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                       ha="center", va="center",
                       fontsize=10.5, fontweight="bold", color=txt_c)
    ax_cm.set_xticks(range(n))
    ax_cm.set_yticks(range(n))
    ax_cm.set_xticklabels([c.capitalize() for c in classes], fontsize=9.5)
    ax_cm.set_yticklabels([c.capitalize() for c in classes], fontsize=9.5)
    ax_cm.set_xlabel("Prediksi", fontsize=10)
    ax_cm.set_ylabel("Aktual", fontsize=10)
    ax_cm.set_title("Confusion Matrix (Test Set)",
                    fontsize=11, fontweight="bold", color="#212529")

    # -- F1 PER KELAS BAR --------------------------------------------------
    ax_f1 = fig.add_subplot(gs[1, 2:4])
    ax_f1.set_facecolor(VIZ_BG)
    x = np.arange(n)
    w = 0.25
    bar_f = ax_f1.bar(x - w, f1s,  w, color=CLASS_COLORS,
                      alpha=0.85, edgecolor="white", label="F1")
    bar_p = ax_f1.bar(x,     prcs, w, color=CLASS_COLORS,
                      alpha=0.55, edgecolor="white", label="Precision",
                      hatch="//")
    bar_r = ax_f1.bar(x + w, recs, w, color=CLASS_COLORS,
                      alpha=0.40, edgecolor="white", label="Recall",
                      hatch="xx")
    for bars in [bar_f, bar_p, bar_r]:
        for bar in bars:
            ax_f1.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.012,
                       f"{bar.get_height()*100:.0f}%",
                       ha="center", va="bottom", fontsize=7.5,
                       fontweight="bold")
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
    ax_f1.set_ylim(0, 1.18)
    ax_f1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_f1.set_title("F1 / Precision / Recall per Kelas (Test Set)",
                    fontsize=11, fontweight="bold", color="#212529")
    ax_f1.legend(fontsize=8, framealpha=0.85)
    ax_f1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax_f1.spines[["top", "right"]].set_visible(False)

    # -- REKOMENDASI LANGKAH SELANJUTNYA -----------------------------------
    ax_rec = fig.add_subplot(gs[2, :])
    ax_rec.set_facecolor("#ffffff")
    for sp in ax_rec.spines.values():
        sp.set_visible(True)
        sp.set_color("#dee2e6")
    ax_rec.axis("off")

    worst_cls_idx  = int(np.argmin(f1s))
    worst_cls_name = classes[worst_cls_idx].capitalize()
    worst_f1       = f1s[worst_cls_idx] * 100
    gap_f1         = (metrics["f1_macro"] - CV_BEST_F1_MACRO) * 100

    if gap_f1 < -5:
        gen_status = "[ERR] OVERFIT TERDETEKSI"
        gen_color  = "#e74c3c"
    elif abs(gap_f1) <= 5:
        gen_status = "[OK] Generalisasi Normal"
        gen_color  = "#27ae60"
    else:
        gen_status = "[WARN] Test > CV (Periksa split)"
        gen_color  = "#f39c12"

    recs_text = [
        (f"STATUS: {gen_status}  |  "
         f"F1-Macro Test: {metrics['f1_macro']*100:.2f}%  |  "
         f"Generalization Gap: {gap_f1:+.2f}%",
         gen_color, 12, True),
        ("", "#aaaaaa", 6, False),
        (f"ANALISIS: Kelas terlemah = {worst_cls_name} (F1={worst_f1:.1f}%). "
         f"Macro AUC={macro_auc:.4f}.",
         "#34495e", 11, False),
        ("", "#aaaaaa", 4, False),
        ("REKOMENDASI LANGKAH SELANJUTNYA (berurutan):", "#212529", 11, True),
    ]

    if worst_f1 < 50:
        recs_text += [
            (f"  1.  SMOTE: Neutral hanya 5.5% data - coba SMOTE pada X_train_emb.npy "
             f"-> retrain SVM -> compare F1-Macro.",
             "#2980b9", 10, False),
            ("  2.  Perluasan grid C: tambah C=1000 (best ada di C=1, mungkin belum optimal).",
             "#2980b9", 10, False),
            ("  3.  Ganti embedding: IndoBERT/IndoBERTweet embedding + SVM "
             f"(literature: ~75-82% F1 untuk Indo sentiment).",
             "#8e44ad", 10, False),
            ("  4.  Audit label: jika masih mentok setelah embedding baru, "
             "periksa kualitas label weak-supervised.",
             "#e74c3c", 10, False),
        ]
    elif worst_f1 < 70:
        recs_text += [
            ("  1.  Coba SMOTE terlebih dahulu - potensi gain 2-5% F1-Macro.",
             "#2980b9", 10, False),
            ("  2.  Perluasan grid C -> C=1000.",
             "#2980b9", 10, False),
            ("  3.  Pertimbangkan IndoBERT embedding untuk gain lebih besar.",
             "#8e44ad", 10, False),
        ]
    else:
        recs_text += [
            ("  [OK] Performa per kelas sudah cukup seimbang.",
             "#27ae60", 10, False),
            ("  -> Pertimbangkan fine-tuning threshold untuk aplikasi produksi.",
             "#2980b9", 10, False),
        ]

    y_pos = 0.93
    for text, color, size, bold in recs_text:
        if not text:
            y_pos -= 0.04
            continue
        ax_rec.text(0.015, y_pos, text,
                    transform=ax_rec.transAxes,
                    fontsize=size, color=color,
                    fontweight="bold" if bold else "normal",
                    va="top")
        y_pos -= (size * 0.012)

    fig.suptitle(
        f"[DASHBOARD] DASHBOARD EVALUASI FINAL - SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA}\n"
        f"Sentiment Analysis Indonesian Reviews | "
        f"Test Set: {cm.sum()} sampel | 3 Kelas",
        fontsize=14, fontweight="bold", color="#212529", y=1.01)

    path = os.path.join(output_dir, "viz_eval_summary_dashboard.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_eval_summary_dashboard.png ({os.path.getsize(path)//1024} KB) <- UNGGULAN")
    return path


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    global log_lines
    t_start   = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    banner = (
        "\n" + "=" * 65 + "\n"
        "  EVALUASI FINAL - SVM + SBERT Sentiment Analysis\n"
        f"  Start     : {timestamp}\n"
        "  TAHAP INI : Evaluasi generalization pada X_test\n"
        "              (X_test disentuh PERTAMA KALI di sini!)\n"
        f"  Model     : SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA}\n"
        f"  CV F1-Macro (dari tahap Modeling): {CV_BEST_F1_MACRO*100:.4f}%\n"
        + "=" * 65
    )
    print(C_HEAD + banner + C_RST)
    log_lines.append(banner)

    # =========================================================================
    # TAHAP A: VERIFIKASI FILE
    # =========================================================================
    print_section("TAHAP A: VERIFIKASI FILE INPUT")
    required = [
        (IN_BEST_MODEL, "svm_sbert_best_model.joblib"),
        (IN_X_TEST,     "X_test_emb.npy"),
        (IN_Y_TEST,     "y_test.npy"),
        (IN_CLASSES,    "label_classes.npy"),
    ]
    for path, name in required:
        if not os.path.exists(path):
            log_err(f"File tidak ditemukan: {path}")
            log_err("Pastikan tahap Modeling + Feature Extraction sudah dijalankan.")
            return
        log_ok(f"{name} ditemukan ({os.path.getsize(path)//1024} KB)")

    # =========================================================================
    # TAHAP B: LOAD
    # =========================================================================
    print_section("TAHAP B: LOAD MODEL + TEST DATA")

    model   = joblib.load(IN_BEST_MODEL)
    X_test  = np.load(IN_X_TEST).astype(np.float32)
    y_test  = np.load(IN_Y_TEST)
    classes = np.load(IN_CLASSES, allow_pickle=True).tolist()
    n_test  = X_test.shape[0]
    n_cls   = len(classes)

    log_ok(f"Model loaded: {type(model).__name__}")
    log_ok(f"X_test : {X_test.shape}  dtype={X_test.dtype}")
    log_ok(f"y_test : {y_test.shape}")
    log_ok(f"Classes: {classes}")

    # Distribusi kelas test
    log("\n  [INFO] Distribusi kelas test set:")
    for i, cls in enumerate(classes):
        cnt = int((y_test == i).sum())
        log(f"    {i} -> {cls:<12} | {cnt:>4} sampel ({cnt/n_test*100:.1f}%)")

    # Verifikasi model punya probability
    has_proba = hasattr(model, "predict_proba")
    if not has_proba:
        log_warn("Model tidak punya predict_proba! ROC/PR/Calibration tidak bisa dibuat.")
        log_warn("Pastikan model disimpan dengan probability=True.")

    # =========================================================================
    # TAHAP C: PREDIKSI
    # =========================================================================
    print_section("TAHAP C: PREDIKSI PADA TEST SET")

    t0     = time.time()
    y_pred = model.predict(X_test)
    dur_pred = time.time() - t0
    log_ok(f"Prediksi selesai: {n_test} sampel dalam {dur_pred:.3f}s "
           f"({n_test/dur_pred:.0f} sampel/detik)")

    y_proba = None
    if has_proba:
        y_proba = model.predict_proba(X_test)
        log_ok(f"Probabilitas: {y_proba.shape}  "
               f"mean max_prob = {y_proba.max(axis=1).mean():.4f}")

    # =========================================================================
    # TAHAP D: HITUNG METRIK
    # =========================================================================
    print_section("TAHAP D: KALKULASI METRIK EVALUASI FINAL")

    metrics = {
        "accuracy"        : accuracy_score(y_test, y_pred),
        "f1_macro"        : f1_score(y_test, y_pred, average="macro",    zero_division=0),
        "f1_weighted"     : f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_micro"        : f1_score(y_test, y_pred, average="micro",    zero_division=0),
        "precision_macro" : precision_score(y_test, y_pred, average="macro",    zero_division=0),
        "recall_macro"    : recall_score(y_test,    y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted" : recall_score(y_test,    y_pred, average="weighted", zero_division=0),
    }

    # ROC AUC (jika ada proba)
    macro_auc  = 0.0
    auc_scores = [0.0] * n_cls
    ap_scores  = [0.0] * n_cls
    mean_ap    = 0.0

    if y_proba is not None:
        try:
            y_bin = label_binarize(y_test, classes=list(range(n_cls)))
            macro_auc = roc_auc_score(y_bin, y_proba, average="macro",
                                      multi_class="ovr")
            metrics["roc_auc_macro"] = macro_auc
            for i in range(n_cls):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc_scores[i] = auc(fpr_i, tpr_i)
                ap_scores[i]  = average_precision_score(y_bin[:, i], y_proba[:, i])
            mean_ap = np.mean(ap_scores)
            metrics["mean_average_precision"] = mean_ap
        except Exception as e:
            log_warn(f"ROC AUC gagal dihitung: {e}")

    cm = confusion_matrix(y_test, y_pred)
    f1s  = f1_score(y_test, y_pred, average=None, zero_division=0)
    prcs = precision_score(y_test, y_pred, average=None, zero_division=0)
    recs = recall_score(y_test, y_pred, average=None, zero_division=0)

    # -- Classification Report ----------------------------------------------
    cr = classification_report(y_test, y_pred,
                                target_names=[c.capitalize() for c in classes],
                                zero_division=0)
    cr_path = os.path.join(OUTPUT_DIR, "eval_classification_report.txt")
    with open(cr_path, "w", encoding="utf-8") as f:
        header = (f"CLASSIFICATION REPORT - SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA}\n"
                  f"Test Set Evaluasi Final - {timestamp}\n"
                  f"{'='*60}\n\n")
        f.write(header + cr)
        f.write(f"\n\nCV F1-Macro (dari tahap Modeling): {CV_BEST_F1_MACRO*100:.4f}%\n")
        f.write(f"Test F1-Macro                    : {metrics['f1_macro']*100:.4f}%\n")
        f.write(f"Generalization Gap               : "
                f"{(metrics['f1_macro']-CV_BEST_F1_MACRO)*100:+.4f}%\n")
    log_ok(f"eval_classification_report.txt tersimpan")

    # -- Print ringkasan ----------------------------------------------------
    log("\n" + "-" * 50)
    log("  HASIL EVALUASI TEST SET FINAL")
    log("-" * 50)
    log(f"  F1-Macro    : {metrics['f1_macro']*100:.4f}%  "
        f"(CV was: {CV_BEST_F1_MACRO*100:.4f}%  "
        f"Gap: {(metrics['f1_macro']-CV_BEST_F1_MACRO)*100:+.4f}%)")
    log(f"  F1-Weighted : {metrics['f1_weighted']*100:.4f}%")
    log(f"  Accuracy    : {metrics['accuracy']*100:.4f}%")
    log(f"  Prec-Macro  : {metrics['precision_macro']*100:.4f}%")
    log(f"  Rec-Macro   : {metrics['recall_macro']*100:.4f}%")
    if macro_auc > 0:
        log(f"  AUC-Macro   : {macro_auc:.4f}")
    log("-" * 50)
    log("\n  Per Kelas:")
    log(f"  {'Kelas':<14} {'F1':>8} {'Precision':>11} {'Recall':>8} "
        f"{'Support':>9} {'AUC':>7} {'AP':>7}")
    log(f"  {'-'*70}")
    for i, cls in enumerate(classes):
        sup = int((y_test == i).sum())
        log(f"  {cls.capitalize():<14} "
            f"{f1s[i]*100:>7.2f}% "
            f"{prcs[i]*100:>10.2f}% "
            f"{recs[i]*100:>7.2f}% "
            f"{sup:>9}"
            f"{auc_scores[i]:>8.4f}"
            f"{ap_scores[i]:>8.4f}")
    log(f"\n  Classification Report:\n{cr}")

    # -- CSV Metrik ---------------------------------------------------------
    rows = [{"metrik": k, "nilai": v, "persen": f"{v*100:.4f}%"}
            for k, v in metrics.items()]
    rows.append({"metrik": "cv_f1_macro_training",
                 "nilai": CV_BEST_F1_MACRO,
                 "persen": f"{CV_BEST_F1_MACRO*100:.4f}%"})
    rows.append({"metrik": "generalization_gap",
                 "nilai": metrics["f1_macro"] - CV_BEST_F1_MACRO,
                 "persen": f"{(metrics['f1_macro']-CV_BEST_F1_MACRO)*100:+.4f}%"})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "eval_metrics_summary.csv"), index=False)
    log_ok("eval_metrics_summary.csv tersimpan")

    # =========================================================================
    # TAHAP E: VISUALISASI
    # =========================================================================
    print_section("TAHAP E: MEMBUAT 9 VISUALISASI")

    viz_confusion_matrix_test(cm, classes, metrics, OUTPUT_DIR)
    _, f1s, prcs, recs = viz_per_class_metrics(y_test, y_pred, classes, OUTPUT_DIR)
    viz_cv_vs_test_gap(metrics, OUTPUT_DIR)

    if y_proba is not None:
        _, macro_auc, auc_scores = viz_roc_multiclass(
            y_test, y_proba, classes, OUTPUT_DIR)
        _, mean_ap, ap_scores = viz_pr_curve(
            y_test, y_proba, classes, OUTPUT_DIR)
        viz_confidence_dist(y_test, y_pred, y_proba, classes, OUTPUT_DIR)
        viz_calibration(y_test, y_proba, classes, OUTPUT_DIR)
    else:
        log_warn("Probabilitas tidak tersedia - skip viz ROC/PR/Confidence/Calibration")

    viz_error_analysis(cm, classes, OUTPUT_DIR)

    viz_summary_dashboard(
        metrics, cm, classes, f1s, prcs, recs,
        macro_auc, ap_scores, OUTPUT_DIR)

    # =========================================================================
    # TAHAP F: LOG PENUTUP + REKOMENDASI
    # =========================================================================
    elapsed = time.time() - t_start
    th, rem = divmod(int(elapsed), 3600)
    tm, ts  = divmod(rem, 60)

    worst_idx  = int(np.argmin(f1s))
    worst_name = classes[worst_idx]
    worst_f1   = f1s[worst_idx] * 100
    gap_f1     = (metrics["f1_macro"] - CV_BEST_F1_MACRO) * 100

    closing = (
        "\n" + "=" * 65 + "\n"
        "  EVALUASI FINAL SELESAI!\n"
        f"\n  MODEL: SVM {BEST_KERNEL} C={BEST_C} gamma={BEST_GAMMA}\n"
        f"\n  HASIL TEST SET:\n"
        f"    F1-Macro    : {metrics['f1_macro']*100:.4f}%\n"
        f"    F1-Weighted : {metrics['f1_weighted']*100:.4f}%\n"
        f"    Accuracy    : {metrics['accuracy']*100:.4f}%\n"
        f"    AUC-Macro   : {macro_auc:.4f}\n"
        f"\n  GENERALISASI:\n"
        f"    CV F1-Macro  : {CV_BEST_F1_MACRO*100:.4f}%\n"
        f"    Test F1-Macro: {metrics['f1_macro']*100:.4f}%\n"
        f"    Gap          : {gap_f1:+.4f}%  "
        f"({'OK' if abs(gap_f1) <= 5 else 'PERLU DIPERIKSA'})\n"
        f"\n  PER KELAS:\n"
        f"    {'Kelas':<12} {'F1':>8} {'P':>8} {'R':>8}\n"
    )
    for i, cls in enumerate(classes):
        closing += (f"    {cls.capitalize():<12} "
                    f"{f1s[i]*100:>7.2f}%"
                    f"{prcs[i]*100:>7.2f}%"
                    f"{recs[i]*100:>7.2f}%\n")
    closing += (
        f"\n  KELAS TERLEMAH: {worst_name.upper()} (F1={worst_f1:.1f}%)\n"
        f"\n  REKOMENDASI SELANJUTNYA:\n"
        f"    1. Jika Neutral F1 < 50%: coba SMOTE pada X_train_emb\n"
        f"    2. Perluas grid: tambah C=1000\n"
        f"    3. Ganti embedding ke IndoBERT/IndoBERTweet\n"
        f"    4. Audit label jika masih mentok\n"
        f"\n  OUTPUT:\n"
        f"    eval_metrics_summary.csv\n"
        f"    eval_classification_report.txt\n"
        f"    viz_eval_confusion_matrix.png\n"
        f"    viz_eval_per_class_metrics.png\n"
        f"    viz_eval_cv_vs_test_gap.png\n"
        f"    viz_eval_roc_multiclass.png\n"
        f"    viz_eval_pr_curve.png\n"
        f"    viz_eval_confidence_dist.png\n"
        f"    viz_eval_calibration.png\n"
        f"    viz_eval_error_analysis.png\n"
        f"    viz_eval_summary_dashboard.png   <- UNGGULAN\n"
        f"    evaluation_final_log.txt\n"
        f"\n  Waktu total: {th}h {tm}m {ts:02d}s\n"
        + "=" * 65
    )
    print(C_HEAD + closing + C_RST)
    log_lines.append(closing)

    log_path = os.path.join(OUTPUT_DIR, "evaluation_final_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log_ok(f"Log tersimpan: {log_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
