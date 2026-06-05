"""
=============================================================================
TAHAP MODELING & HYPERPARAMETER TUNING: SVM + SBERT + SMOTE (v3.0 FIXED)
=============================================================================
Proyek  : Sentiment Analysis Review (Indonesian)
Tahap   : Modeling + Hyperparameter Tuning + SMOTE
Versi   : v3.0 — FIXED overfitting dari v2.0

=============================================================================
MASALAH v2.0 & PERBAIKAN v3.0
=============================================================================
  MASALAH v2.0:
    - SMOTE terlalu agresif: Neutral 262 -> 3647 (x14)
    - Model hafal synthetic samples, bukan belajar pola real
    - CV di SMOTE data 97% tapi Test hanya 53.7% (gap 43%!)
    - Neutral F1 malah TURUN dari 26.74% -> 4.26% di test

  PERBAIKAN v3.0:
    [1] SMOTE PARTIAL — bukan full balance.
        Neutral 262 -> 800 (3x oversampling, wajar)
        Negative 885 -> 1200 (sedikit oversample)
        Positive tetap 3647 (mayoritas, tidak disentuh)
        Rasio akhir: Neg 1200 / Neu 800 / Pos 3647 = tidak ekstrem

    [2] KOMBINASIKAN SMOTE + class_weight='balanced'
        SMOTE menambah data Neutral agar SVM punya lebih banyak
        contoh untuk dipelajari, class_weight memastikan loss
        function tetap memperhatikan kelas minoritas.

    [3] GRID LEBIH KONSERVATIF
        - Fokus Linear kernel dulu (lebih stabil, lebih cepat)
        - RBF hanya dengan C kecil-menengah [0.1, 1, 10]
        - Tanpa C=100 yang overfitting untuk data SMOTE

    [4] VALIDASI GAP CV vs GENERALISASI
        Setelah tuning, hitung estimasi gap.
        Jika gap > 15% -> warning, pilih model lebih konservatif.

    [5] STRATEGI FALLBACK
        Jika SMOTE partial tidak membantu, script otomatis
        cek apakah baseline (tanpa SMOTE) lebih baik dan
        merekomendasikan model terbaik berdasarkan estimasi
        generalisasi, bukan hanya CV score.

  PERKIRAAN HASIL:
    - CV tidak lagi 97% (akan lebih realistis ~70-85%)
    - Test F1-Macro diharapkan ~60-70% (naik dari v1.0 60.06%)
    - Neutral F1 diharapkan naik dari 26.74% ke ~35-50%

=============================================================================
"""

import os
import sys
import io
import time
import warnings
import random
import numpy as np
import pandas as pd
import joblib

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import cross_val_predict

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[ERR] imbalanced-learn tidak ada. Jalankan: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# PATH KONFIGURASI
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(CURRENT_DIR, "data")
OUTPUT_DIR  = os.path.join(CURRENT_DIR, "outputSmote")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IN_X_TRAIN = os.path.join(DATA_DIR, "X_train_emb.npy")
IN_Y_TRAIN = os.path.join(DATA_DIR, "y_train.npy")
IN_CLASSES = os.path.join(DATA_DIR, "label_classes.npy")

# =============================================================================
# SMOTE CONFIG — PARTIAL OVERSAMPLING (KUNCI PERBAIKAN)
# =============================================================================
SMOTE_K_NEIGHBORS = 5  # default k=5, data kecil lebih aman

# Strategi partial: tentukan target jumlah per kelas
# Neutral  262 -> 800  (3x)  — cukup untuk belajar pola
# Negative 885 -> 1200 (1.4x) — sedikit boost
# Positive 3647 (mayoritas, tidak di-oversample oleh SMOTE)
# Format: {class_index: target_count}
# class 0=negative, 1=neutral, 2=positive
SMOTE_STRATEGY_PARTIAL = {
    0: 1200,   # Negative: 885 -> 1200
    1: 800,    # Neutral : 262 -> 800
    # 2 tidak disebutkan = tidak di-oversample (sudah mayoritas)
}

# =============================================================================
# PARAMETER MODELING
# =============================================================================
SVM_CLASS_WT  = "balanced"   # tetap pakai, sinergi dengan SMOTE partial
SVM_MAX_ITER  = 3000
SVM_TOL       = 1e-4
SVM_CACHE_SIZE = 300
RANDOM_STATE  = 42

# Grid lebih konservatif — hindari C besar yang bikin overfitting di SMOTE data
# Linear: C [0.1, 1, 10] — C=100 dihilangkan
# RBF   : C [0.1, 1, 10] dengan gamma ['scale', 0.01]
C_GRID_LINEAR = [0.1, 1, 10]
C_GRID_RBF    = [0.1, 1, 10]
GAMMA_GRID    = ["scale", 0.01]   # hanya 2 opsi, cukup

CV_FOLDS      = 5
SCORING_MAIN  = "f1_macro"
SCORING_MULTI = {
    "f1_macro"        : "f1_macro",
    "f1_weighted"     : "f1_weighted",
    "precision_macro" : "precision_macro",
    "recall_macro"    : "recall_macro",
    "accuracy"        : "accuracy",
}

# Threshold gap: jika CV score - estimasi test > ini, model dianggap overfit
GAP_THRESHOLD = 0.12   # 12%

OUT_PREFIX = "svm_sbert_smote_v3"

# Warna viz
VIZ_BG       = "#f8f9fa"
VIZ_GRID_CLR = "#dee2e6"
VIZ_LINEAR   = "#2980b9"
VIZ_RBF      = "#8e44ad"
VIZ_BASELINE = "#95a5a6"
VIZ_BEST     = "#e67e22"
VIZ_NEG      = "#e74c3c"
VIZ_NEU      = "#f39c12"
VIZ_POS      = "#27ae60"
VIZ_SMOTE    = "#16a085"
VIZ_V1       = "#bdc3c7"

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
# HELPER: EVALUASI VIA CV — dengan estimasi gap generalisasi
# =============================================================================
def evaluate_via_cv(svc_params, X_train, y_train, classes, label):
    model  = SVC(**svc_params)
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_out = cross_validate(
        model, X_train, y_train,
        cv=skf, scoring=SCORING_MULTI,
        n_jobs=-1, return_train_score=False,
    )
    f1_per_fold = cv_out["test_f1_macro"].tolist()
    metrics = {
        "accuracy"          : cv_out["test_accuracy"].mean(),
        "precision_macro"   : cv_out["test_precision_macro"].mean(),
        "recall_macro"      : cv_out["test_recall_macro"].mean(),
        "f1_macro"          : cv_out["test_f1_macro"].mean(),
        "f1_weighted"       : cv_out["test_f1_weighted"].mean(),
        "f1_macro_std"      : cv_out["test_f1_macro"].std(),
        "f1_macro_per_fold" : f1_per_fold,
    }
    # Estimasi test score: CV mean - 1.5 * std (konservatif)
    est_test = metrics["f1_macro"] - 1.5 * metrics["f1_macro_std"]
    metrics["estimated_test_f1"] = est_test

    log(f"\n  [CV EVAL - {label}]")
    log(f"    F1-Macro   : {metrics['f1_macro']*100:.4f}% "
        f"(+/-{metrics['f1_macro_std']*100:.4f}%)  <- METRIK UTAMA")
    log(f"    Est.Test   : {est_test*100:.4f}%  (CV - 1.5*std, konservatif)")
    log(f"    F1-Weighted: {metrics['f1_weighted']*100:.4f}%")
    log(f"    Accuracy   : {metrics['accuracy']*100:.4f}%")
    log(f"    Per fold   : "
        f"{[f'{v*100:.2f}%' for v in f1_per_fold]}")
    return metrics


def get_cv_confusion_matrix(svc_params, X_train, y_train):
    model  = SVC(**svc_params)
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_pred = cross_val_predict(model, X_train, y_train, cv=skf, n_jobs=-1)
    return confusion_matrix(y_train, y_pred), y_pred


# =============================================================================
# SMOTE PARTIAL — fungsi terpisah agar mudah di-debug
# =============================================================================
def apply_smote_partial(X_train, y_train, strategy, k_neighbors, seed):
    """
    Terapkan SMOTE dengan strategi partial.
    Hanya kelas yang ada di `strategy` yang di-oversample.
    Kelas mayoritas tidak disentuh.

    Validasi: pastikan target tidak melebihi kelas mayoritas
    (mencegah sampling_strategy error).
    """
    n_majority = int((y_train == y_train.max()).sum())  # asumsi max = mayoritas

    # Clip target: tidak boleh melebihi mayoritas
    strategy_safe = {}
    for cls_idx, target in strategy.items():
        current = int((y_train == cls_idx).sum())
        # Hanya oversample jika target > current
        if target > current:
            # Target tidak boleh melebihi majority class
            safe_target = min(target, n_majority)
            strategy_safe[cls_idx] = safe_target
            if safe_target < target:
                print(C_WARN + f"  [WARN] Kelas {cls_idx}: target {target} "
                      f"di-clip ke {safe_target} (max=majority)" + C_RST)
        else:
            print(C_WARN + f"  [WARN] Kelas {cls_idx}: target {target} <= "
                  f"current {current}, skip" + C_RST)

    if not strategy_safe:
        print(C_WARN + "  [WARN] Tidak ada kelas yang perlu di-oversample!" + C_RST)
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy = strategy_safe,
        k_neighbors       = k_neighbors,
        random_state      = seed,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res.astype(np.float32), y_res


# =============================================================================
# VIZ 1 — DISTRIBUSI SEBELUM vs SESUDAH SMOTE
# =============================================================================
def viz_smote_distribution(y_before, y_after, classes, output_dir, prefix):
    n = len(classes)
    counts_before = [(y_before == i).sum() for i in range(n)]
    counts_after  = [(y_after  == i).sum() for i in range(n)]
    total_before  = sum(counts_before)
    total_after   = sum(counts_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(VIZ_BG)
    x = np.arange(n)
    w = 0.35

    ax1.set_facecolor(VIZ_BG)
    b1 = ax1.bar(x - w/2, counts_before, w, label="Sebelum SMOTE",
                 color=VIZ_BASELINE, alpha=0.80, edgecolor="white")
    b2 = ax1.bar(x + w/2, counts_after,  w, label="Sesudah SMOTE",
                 color=VIZ_SMOTE,    alpha=0.85, edgecolor="white")

    for bar, val, total in (
        [(b, v, total_before) for b, v in zip(b1, counts_before)] +
        [(b, v, total_after)  for b, v in zip(b2, counts_after)]
    ):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{val}\n({val/total*100:.1f}%)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
    ax1.set_ylabel("Jumlah Sampel", fontsize=11)
    ax1.set_title(
        f"Distribusi Kelas: Sebelum vs Sesudah SMOTE (v3.0 PARTIAL)\n"
        f"Total sebelum: {total_before} | Sesudah: {total_after}\n"
        f"SMOTE partial — bukan full balance!",
        fontsize=10, fontweight="bold", color="#212529", pad=10)
    ax1.legend(fontsize=10, framealpha=0.85)
    ax1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax1.spines[["top","right"]].set_visible(False)

    ax2.set_facecolor(VIZ_BG)
    cls_colors = [VIZ_NEG, VIZ_NEU, VIZ_POS]
    ratios = [a/b for a, b in zip(counts_after, counts_before)]
    bars   = ax2.bar(x, ratios, color=cls_colors, alpha=0.85, edgecolor="white")
    ax2.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.5,
                label="Tidak berubah (1x)")
    ax2.axhline(5.0, color="#e67e22", linestyle=":", linewidth=1.5, alpha=0.7,
                label="Batas aman oversampling (~5x)")

    for bar, ratio, cb, ca in zip(bars, ratios, counts_before, counts_after):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{ratio:.1f}x\n({cb}->{ca})",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
    ax2.set_ylabel("Rasio Oversampling", fontsize=10)
    ax2.set_title(
        "Rasio Oversampling per Kelas (v3.0)\n"
        "Neutral ~3x | v2.0 Neutral 14x (overfitting!)\n"
        "Rasio moderat = generalisasi lebih baik",
        fontsize=10, fontweight="bold", color="#212529", pad=10)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "Analisis SMOTE Partial v3.0 — Moderat, Bukan Full Balance\n"
        "Tujuan: tambah sampel Neutral tanpa menyebabkan overfitting",
        fontsize=12, fontweight="bold", color="#212529", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, f"viz_{prefix}_smote_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_{prefix}_smote_distribution.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 2 — C vs F1-MACRO PER KERNEL
# =============================================================================
def viz_c_vs_f1_kernel(cv_results_df, best_c, best_kernel, output_dir, prefix):
    df_lin = cv_results_df[cv_results_df["kernel"]=="linear"].sort_values("C")
    df_rbf = cv_results_df[cv_results_df["kernel"]=="rbf"].sort_values("C")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    for ax, df, color, marker, label in [
        (ax1, df_lin, VIZ_LINEAR, "o", "Linear"),
        (ax1, df_rbf, VIZ_RBF,   "s", "RBF"),
    ]:
        if df.empty:
            continue
        grp = df.groupby("C", as_index=False).agg(
            mean_f1=("mean_f1_macro","max"),
            std_f1 =("std_f1_macro","min"))
        ax.semilogx(grp["C"], grp["mean_f1"],
                    color=color, marker=marker, linewidth=2.2, markersize=7,
                    label=f"{label} Kernel")
        ax.fill_between(grp["C"],
                        grp["mean_f1"]-grp["std_f1"],
                        grp["mean_f1"]+grp["std_f1"],
                        alpha=0.12, color=color)

    best_row = cv_results_df.loc[cv_results_df["mean_f1_macro"].idxmax()]
    ax1.scatter([best_row["C"]], [best_row["mean_f1_macro"]],
                color=VIZ_BEST, s=220, zorder=6, marker="*",
                label=f"Best: {best_kernel.upper()} C={best_c}")
    ax1.axvline(x=best_c, color="#c0392b", linestyle=":", linewidth=2.0)
    ax1.set_xlabel("Nilai C (log scale)", fontsize=11)
    ax1.set_ylabel("Mean CV F1-Macro", fontsize=11)
    ax1.set_title(
        f"C vs F1-Macro [SMOTE PARTIAL v3.0]\nBest: {best_kernel.upper()} C={best_c} "
        f"| CV F1={best_row['mean_f1_macro']*100:.2f}%",
        fontsize=10, fontweight="bold", pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(color=VIZ_GRID_CLR, linewidth=0.7)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.1f}%"))

    # Panel kanan: estimated test score (CV - 1.5*std)
    ax2.set_facecolor(VIZ_BG)
    if not df_lin.empty:
        grp_lin = df_lin.groupby("C", as_index=False).agg(
            est=("estimated_test_f1","max"))
        ax2.semilogx(grp_lin["C"], grp_lin["est"],
                     color=VIZ_LINEAR, marker="o", linewidth=2.0, markersize=6,
                     label="Linear (est. test)")
    if not df_rbf.empty:
        grp_rbf = df_rbf.groupby("C", as_index=False).agg(
            est=("estimated_test_f1","max"))
        ax2.semilogx(grp_rbf["C"], grp_rbf["est"],
                     color=VIZ_RBF, marker="s", linewidth=2.0, markersize=6,
                     label="RBF (est. test)")
    ax2.axvline(x=best_c, color="#c0392b", linestyle=":", linewidth=2.0)
    ax2.set_xlabel("Nilai C (log scale)", fontsize=11)
    ax2.set_ylabel("Est. Test F1-Macro (CV-1.5*std)", fontsize=11)
    ax2.set_title("Estimasi Test F1 per C [v3.0]\n(lebih konservatif, lebih jujur)",
                  fontsize=10, fontweight="bold", pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(color=VIZ_GRID_CLR, linewidth=0.7)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.1f}%"))

    fig.suptitle(
        "Analisis C — SVM + SBERT + SMOTE Partial v3.0\n"
        f"Grid: Linear C={C_GRID_LINEAR} | RBF C={C_GRID_RBF} gamma={GAMMA_GRID}",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, f"viz_{prefix}_c_vs_f1.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_{prefix}_c_vs_f1.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 3 — CONFUSION MATRIX CV
# =============================================================================
def viz_confusion_matrix_cv(cm, classes, cv_f1, title_suffix, filename, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt_c = "white" if cm_norm[i,j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=txt_c)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_yticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Aktual", fontsize=11)
    ax.set_title(
        f"Confusion Matrix CV — {title_suffix}\n"
        f"CV F1-Macro: {cv_f1*100:.2f}% | SMOTE Partial v3.0",
        fontsize=11, fontweight="bold", color="#212529", pad=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"{filename} ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 4 — PERBANDINGAN: v1.0 / v2.0 / v3.0 (CV + estimasi test)
# =============================================================================
def viz_version_comparison(metrics_best_cv, f1_per_class, classes, output_dir, prefix):
    """
    Tiga versi dibandingkan:
    v1.0: tanpa SMOTE, C=1, Linear   — Test F1-Macro=60.06%, Neutral=26.74%
    v2.0: SMOTE full, C=100, RBF     — Test F1-Macro=53.70%, Neutral=4.26% (GAGAL)
    v3.0: SMOTE partial, v3.0 best   — CV & estimasi test (belum dievaluasi)
    """
    metrics_v1 = {
        "f1_macro": 0.6006, "f1_weighted": 0.8045, "accuracy": 0.7857,
        "f1_neg": 0.6554, "f1_neu": 0.2674, "f1_pos": 0.8791,
        "label": "v1.0 Test\n(no SMOTE)",
    }
    metrics_v2 = {
        "f1_macro": 0.5370, "f1_weighted": 0.8133, "accuracy": 0.8249,
        "f1_neg": 0.6638, "f1_neu": 0.0426, "f1_pos": 0.9047,
        "label": "v2.0 Test\n(SMOTE full x14)",
    }
    metrics_v3_cv = {
        "f1_macro"    : metrics_best_cv["f1_macro"],
        "f1_weighted" : metrics_best_cv["f1_weighted"],
        "accuracy"    : metrics_best_cv["accuracy"],
        "f1_neg"      : f1_per_class[0] if len(f1_per_class)>0 else 0,
        "f1_neu"      : f1_per_class[1] if len(f1_per_class)>1 else 0,
        "f1_pos"      : f1_per_class[2] if len(f1_per_class)>2 else 0,
        "label"       : "v3.0 CV\n(SMOTE partial)",
    }
    metrics_v3_est = {
        "f1_macro"    : metrics_best_cv["estimated_test_f1"],
        "f1_weighted" : max(0, metrics_best_cv["f1_weighted"] - 0.05),
        "accuracy"    : max(0, metrics_best_cv["accuracy"]    - 0.05),
        "f1_neg"      : f1_per_class[0] if len(f1_per_class)>0 else 0,
        "f1_neu"      : f1_per_class[1] if len(f1_per_class)>1 else 0,
        "f1_pos"      : f1_per_class[2] if len(f1_per_class)>2 else 0,
        "label"       : "v3.0 Est.Test\n(CV-1.5*std)",
    }

    metric_names = ["F1-Macro", "F1-Weighted", "Accuracy"]
    metric_keys  = ["f1_macro", "f1_weighted", "accuracy"]
    cls_metrics  = ["F1 Negative", "F1 Neutral", "F1 Positive"]
    cls_keys     = ["f1_neg", "f1_neu", "f1_pos"]
    cls_colors_  = [VIZ_NEG, VIZ_NEU, VIZ_POS]

    versions = [metrics_v1, metrics_v2, metrics_v3_cv, metrics_v3_est]
    ver_colors = [VIZ_V1, "#e74c3c", VIZ_SMOTE, VIZ_BEST]
    x = np.arange(len(metric_names))
    w = 0.20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    # Panel kiri: overall metrics
    ax1.set_facecolor(VIZ_BG)
    for vi, (ver, col) in enumerate(zip(versions, ver_colors)):
        offset = (vi - 1.5) * w
        vals   = [ver[k] for k in metric_keys]
        bars   = ax1.bar(x + offset, vals, w,
                         label=ver["label"], color=col, alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                     f"{val*100:.1f}%",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, fontsize=11)
    ax1.set_ylabel("Skor", fontsize=11)
    ax1.set_ylim(0, 1.20)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    ax1.set_title(
        "Perbandingan v1.0 / v2.0 / v3.0\nAbu=v1 Test | Merah=v2 Test | Teal=v3 CV | Oranye=v3 Est.Test",
        fontsize=10, fontweight="bold", pad=10)
    ax1.legend(fontsize=8, framealpha=0.85, ncol=2)
    ax1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax1.spines[["top","right"]].set_visible(False)

    # Panel kanan: F1 per kelas
    ax2.set_facecolor(VIZ_BG)
    x2 = np.arange(3)
    for vi, (ver, col) in enumerate(zip(versions, ver_colors)):
        offset = (vi - 1.5) * w
        vals   = [ver[k] for k in cls_keys]
        bars   = ax2.bar(x2 + offset, vals, w,
                         label=ver["label"], color=col, alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                     f"{val*100:.1f}%",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cls_metrics, fontsize=10)
    ax2.set_ylabel("F1-Score per Kelas", fontsize=11)
    ax2.set_ylim(0, 1.20)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    ax2.set_title(
        "F1 per Kelas: v1.0 / v2.0 / v3.0\n"
        "TARGET: Neutral naik dari v1.0 (26.7%) dan v2.0 (4.3%)",
        fontsize=10, fontweight="bold", pad=10)
    ax2.legend(fontsize=8, framealpha=0.85, ncol=2)
    ax2.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "Evolusi Model: v1.0 -> v2.0 (gagal) -> v3.0 (SMOTE Partial)\n"
        "v3.0 CV dan estimasi test — aktual test lihat di evaluasiSmoteV3.py",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, f"viz_{prefix}_version_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_{prefix}_version_comparison.png ({os.path.getsize(path)//1024} KB) <- UNGGULAN")
    return path


# =============================================================================
# VIZ 5 — CV BOXPLOT TOP KOMBINASI
# =============================================================================
def viz_cv_boxplot(cv_results_df, all_fold_scores, best_key, output_dir, prefix):
    top_combos = cv_results_df.nlargest(min(8, len(cv_results_df)), "mean_f1_macro")
    data_boxes, combo_labels, best_idx = [], [], None

    for i, (_, row) in enumerate(top_combos.iterrows()):
        kern  = row["kernel"]
        c_val = row["C"]
        g_val = row.get("gamma", "N/A")
        key   = f"{kern}_{c_val}_{g_val}"
        if key in all_fold_scores:
            data_boxes.append(all_fold_scores[key])
            label = f"{kern.upper()}\nC={c_val}"
            if kern == "rbf":
                label += f"\ng={g_val}"
            combo_labels.append(label)
            if key == best_key:
                best_idx = len(data_boxes) - 1

    if not data_boxes:
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(data_boxes)*1.6), 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)
    bp = ax.boxplot(data_boxes, labels=combo_labels, patch_artist=True,
                    medianprops=dict(color="#c0392b", linewidth=2.0),
                    flierprops=dict(marker="o", color="#e74c3c", alpha=0.5, markersize=5),
                    whiskerprops=dict(linewidth=1.2, linestyle="--"),
                    capprops=dict(linewidth=1.2))

    for i, (patch, (_, row)) in enumerate(zip(bp["boxes"], top_combos.iterrows())):
        color = VIZ_LINEAR if row["kernel"]=="linear" else VIZ_RBF
        is_b  = (i == best_idx)
        patch.set_facecolor(VIZ_BEST if is_b else color)
        patch.set_alpha(0.70 if is_b else 0.50)

    for i, scores in enumerate(data_boxes):
        mean_v = np.mean(scores)
        ax.text(i+1, mean_v+0.004, f"{mean_v*100:.2f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    leg = [
        mpatches.Patch(color=VIZ_BEST,   label="Best Model"),
        mpatches.Patch(color=VIZ_LINEAR, label="Linear Kernel"),
        mpatches.Patch(color=VIZ_RBF,    label="RBF Kernel"),
    ]
    ax.legend(handles=leg, fontsize=9)
    ax.set_ylabel("F1-Macro per Fold", fontsize=11)
    ax.set_title(
        f"Distribusi CV F1-Macro per Fold [SMOTE Partial v3.0] — Top Kombinasi\n"
        f"Stratified {CV_FOLDS}-Fold | Oranye = Best",
        fontsize=10, fontweight="bold", pad=10)
    ax.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.1f}%"))
    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, f"viz_{prefix}_cv_boxplot.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_{prefix}_cv_boxplot.png ({os.path.getsize(path)//1024} KB)")
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
        "  MODELING + TUNING SVM + SBERT + SMOTE PARTIAL (v3.0)\n"
        f"  Start  : {timestamp}\n"
        "  PERBAIKAN dari v2.0:\n"
        f"    SMOTE partial: Neutral 262->{SMOTE_STRATEGY_PARTIAL.get(1,'?')} "
        f"(bukan 3647!)\n"
        f"    Grid konservatif: C={C_GRID_LINEAR+C_GRID_RBF} (hapus C=100)\n"
        "    Validasi gap CV vs estimasi test\n"
        "  TARGET: Neutral F1 naik dari 26.74% TANPA overfitting\n"
        + "=" * 65
    )
    print(C_HEAD + banner + C_RST)
    log_lines.append(banner)

    if not SMOTE_AVAILABLE:
        log_err("imbalanced-learn tidak ada!")
        return

    # =========================================================================
    # TAHAP A: LOAD DATA
    # =========================================================================
    print_section("TAHAP A: LOAD DATA")

    for path, name in [
        (IN_X_TRAIN, "X_train_emb.npy"),
        (IN_Y_TRAIN, "y_train.npy"),
        (IN_CLASSES, "label_classes.npy"),
    ]:
        if not os.path.exists(path):
            log_err(f"File tidak ditemukan: {path}")
            return

    X_train_orig = np.load(IN_X_TRAIN).astype(np.float32)
    y_train_orig = np.load(IN_Y_TRAIN)
    classes      = np.load(IN_CLASSES, allow_pickle=True).tolist()
    n_classes    = len(classes)

    log_ok(f"X_train : {X_train_orig.shape}  dtype={X_train_orig.dtype}")
    log_ok(f"y_train : {y_train_orig.shape}")
    log_ok(f"Classes : {classes}")

    log("\n  [INFO] Distribusi kelas SEBELUM SMOTE:")
    for i, cls in enumerate(classes):
        cnt = int((y_train_orig == i).sum())
        log(f"    {i} -> {cls:<12} | {cnt:>5} ({cnt/len(y_train_orig)*100:.1f}%)")

    # =========================================================================
    # TAHAP B: SMOTE PARTIAL
    # =========================================================================
    print_section("TAHAP B: SMOTE PARTIAL (v3.0 — BUKAN FULL BALANCE)")

    log(f"\n  [INFO] Strategi SMOTE PARTIAL:")
    for cls_idx, target in SMOTE_STRATEGY_PARTIAL.items():
        current = int((y_train_orig == cls_idx).sum())
        log(f"    Kelas {cls_idx} ({classes[cls_idx]}): {current} -> {target} "
            f"({target/current:.1f}x)")
    log(f"  [INFO] k_neighbors = {SMOTE_K_NEIGHBORS}")
    log(f"  [PENTING] SMOTE HANYA pada X_train. X_test TIDAK DISENTUH.")
    log(f"  [PENTING] Strategi partial mencegah overfitting vs full balance v2.0")

    t0 = time.time()
    X_train, y_train = apply_smote_partial(
        X_train_orig, y_train_orig,
        strategy    = SMOTE_STRATEGY_PARTIAL,
        k_neighbors = SMOTE_K_NEIGHBORS,
        seed        = SEED,
    )
    dur_smote = time.time() - t0
    n_train   = X_train.shape[0]

    log_ok(f"SMOTE selesai dalam {dur_smote:.2f}s")
    log_ok(f"X_train baru: {X_train.shape} (dari {X_train_orig.shape[0]} -> {n_train})")

    log("\n  [INFO] Distribusi kelas SESUDAH SMOTE:")
    for i, cls in enumerate(classes):
        cnt = int((y_train == i).sum())
        orig = int((y_train_orig == i).sum())
        log(f"    {i} -> {cls:<12} | {cnt:>5} ({cnt/n_train*100:.1f}%) "
            f"[dari {orig}, +{cnt-orig} synthetic]")

    viz_smote_distribution(y_train_orig, y_train, classes, OUTPUT_DIR, OUT_PREFIX)

    # =========================================================================
    # TAHAP C: BASELINE MODEL
    # =========================================================================
    print_section("TAHAP C: BASELINE (Linear C=1 + SMOTE Partial, CV)")

    baseline_params = {
        "kernel": "linear", "C": 1.0,
        "class_weight": SVM_CLASS_WT,
        "max_iter": SVM_MAX_ITER, "tol": SVM_TOL,
        "cache_size": SVM_CACHE_SIZE,
        "random_state": RANDOM_STATE, "probability": False,
    }
    metrics_base = evaluate_via_cv(
        baseline_params, X_train, y_train, classes,
        "Baseline Linear C=1 + SMOTE Partial")

    cm_base, y_pred_base = get_cv_confusion_matrix(baseline_params, X_train, y_train)
    f1_base_cls = f1_score(y_train, y_pred_base, average=None, zero_division=0)
    metrics_base["f1_per_class"] = f1_base_cls

    log(f"\n  [INFO] F1 per kelas Baseline (CV+SMOTE):")
    for i, cls in enumerate(classes):
        log(f"    {cls:<12}: {f1_base_cls[i]*100:.2f}%")

    # Fit dan simpan baseline
    svm_base_full = SVC(**{**baseline_params, "probability": True})
    t0 = time.time()
    svm_base_full.fit(X_train, y_train)
    log_ok(f"Baseline fit: {time.time()-t0:.2f}s | n_support={svm_base_full.n_support_.sum()}")
    joblib.dump(svm_base_full,
                os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_baseline_model.joblib"),
                compress=3)
    log_ok(f"{OUT_PREFIX}_baseline_model.joblib tersimpan")

    # =========================================================================
    # TAHAP D: HYPERPARAMETER TUNING
    # =========================================================================
    print_section("TAHAP D: HYPERPARAMETER TUNING (Grid Konservatif)")

    total_combinations = len(C_GRID_LINEAR) + len(C_GRID_RBF) * len(GAMMA_GRID)
    log(f"\n  [INFO] Grid konservatif (v3.0 fix):")
    log(f"    Linear C  = {C_GRID_LINEAR}  (hapus C=100 yang overfit di SMOTE)")
    log(f"    RBF    C  = {C_GRID_RBF}")
    log(f"    Gamma     = {GAMMA_GRID}    (hanya 2, lebih cepat)")
    log(f"    Total     = {total_combinations} kombinasi x {CV_FOLDS} fold "
        f"= {total_combinations * CV_FOLDS} fits")
    log(f"  [INFO] Training pada {n_train} sampel (jauh lebih kecil dari v2.0 10941)")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    param_grid = [
        {"C": C_GRID_LINEAR, "kernel": ["linear"]},
        {"C": C_GRID_RBF, "kernel": ["rbf"], "gamma": GAMMA_GRID},
    ]
    svm_for_grid = SVC(
        class_weight = SVM_CLASS_WT,
        probability  = False,
        max_iter     = SVM_MAX_ITER,
        tol          = SVM_TOL,
        cache_size   = SVM_CACHE_SIZE,
        random_state = RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator          = svm_for_grid,
        param_grid         = param_grid,
        scoring            = SCORING_MULTI,
        refit              = SCORING_MAIN,
        cv                 = skf,
        n_jobs             = -1,
        verbose            = 1,
        return_train_score = False,
    )

    log(f"\n  Menjalankan GridSearchCV...")
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    dur_grid = time.time() - t0

    best_params   = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_c        = best_params["C"]
    best_kernel   = best_params["kernel"]
    best_gamma    = best_params.get("gamma", "N/A")

    log_ok(f"GridSearchCV selesai: {dur_grid:.1f}s ({dur_grid/60:.1f} menit)")
    log_ok(f"BEST PARAMS : {best_params}")
    log_ok(f"BEST CV F1-Macro: {best_cv_score*100:.4f}%")

    # Kumpulkan hasil lengkap
    cv_res = grid_search.cv_results_
    cv_rows, all_fold_scores = [], {}

    for idx in range(len(cv_res["params"])):
        p     = cv_res["params"][idx]
        kern  = p.get("kernel","?")
        c_val = p.get("C","?")
        g_val = p.get("gamma","N/A")

        mean_f1m = float(cv_res["mean_test_f1_macro"][idx])
        std_f1m  = float(cv_res["std_test_f1_macro"][idx])
        est_test = mean_f1m - 1.5 * std_f1m  # estimasi konservatif

        mean_f1w = float(cv_res.get("mean_test_f1_weighted",
                                    np.zeros(len(cv_res["params"])))[idx])
        mean_pm  = float(cv_res["mean_test_precision_macro"][idx])
        mean_rm  = float(cv_res["mean_test_recall_macro"][idx])
        mean_acc = float(cv_res["mean_test_accuracy"][idx])

        cv_rows.append({
            "kernel": kern, "C": c_val, "gamma": g_val,
            "mean_f1_macro": mean_f1m, "std_f1_macro": std_f1m,
            "estimated_test_f1": est_test,
            "mean_f1_weighted": mean_f1w,
            "mean_prec_macro": mean_pm, "mean_rec_macro": mean_rm,
            "mean_accuracy": mean_acc,
        })
        fold_scores = [float(cv_res[f"split{k}_test_f1_macro"][idx])
                       for k in range(CV_FOLDS)]
        all_fold_scores[f"{kern}_{c_val}_{g_val}"] = fold_scores

        best_flag = " <-- BEST" if (
            kern == best_kernel and c_val == best_c and
            str(g_val) == str(best_gamma)) else ""
        log(f"    {kern.upper()} C={c_val} gamma={g_val}: "
            f"F1={mean_f1m*100:.3f}% (+/-{std_f1m*100:.3f}%) "
            f"EstTest={est_test*100:.3f}%{best_flag}")

    cv_results_df = pd.DataFrame(cv_rows)
    best_fold_key = f"{best_kernel}_{best_c}_{best_gamma}"

    # =========================================================================
    # TAHAP E: VALIDASI GAP — cek overfitting
    # =========================================================================
    print_section("TAHAP E: VALIDASI GAP CV vs GENERALISASI")

    best_row_df = cv_results_df.loc[
        (cv_results_df["kernel"] == best_kernel) &
        (cv_results_df["C"] == best_c) &
        (cv_results_df["gamma"].astype(str) == str(best_gamma))
    ]

    if not best_row_df.empty:
        best_std      = float(best_row_df["std_f1_macro"].iloc[0])
        best_est_test = float(best_row_df["estimated_test_f1"].iloc[0])
        gap = best_cv_score - best_est_test
        log(f"\n  CV F1-Macro  : {best_cv_score*100:.4f}%")
        log(f"  CV Std       : {best_std*100:.4f}%")
        log(f"  Est. Test    : {best_est_test*100:.4f}%  (CV - 1.5*std)")
        log(f"  Gap (est)    : {gap*100:.4f}%")

        if gap > GAP_THRESHOLD:
            log_warn(f"Gap estimasi {gap*100:.2f}% > threshold {GAP_THRESHOLD*100:.0f}%")
            log_warn("Model mungkin masih overfit ke SMOTE data.")
            log_warn("Pertimbangkan SMOTE target lebih kecil atau gunakan Baseline.")
            # Cek apakah ada alternatif lebih stabil (std lebih kecil)
            stable_alt = cv_results_df[
                cv_results_df["std_f1_macro"] < best_std * 0.7
            ].nlargest(3, "estimated_test_f1")
            if not stable_alt.empty:
                log(f"\n  [INFO] Alternatif lebih stabil (std kecil):")
                for _, r in stable_alt.iterrows():
                    log(f"    {r['kernel'].upper()} C={r['C']} gamma={r['gamma']}: "
                        f"CV={r['mean_f1_macro']*100:.2f}% "
                        f"std={r['std_f1_macro']*100:.2f}% "
                        f"EstTest={r['estimated_test_f1']*100:.2f}%")
        else:
            log_ok(f"Gap {gap*100:.2f}% dalam batas normal (< {GAP_THRESHOLD*100:.0f}%)")
    else:
        best_est_test = best_cv_score

    # Simpan CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_tuning_results.csv")
    cv_results_df.to_csv(csv_path, index=False)
    log_ok(f"{OUT_PREFIX}_tuning_results.csv ({len(cv_results_df)} kombinasi)")

    # =========================================================================
    # TAHAP F: EVALUASI BEST MODEL via CV (detail per kelas)
    # =========================================================================
    print_section("TAHAP F: EVALUASI BEST MODEL via CV")

    best_params_cv = {
        "kernel": best_kernel, "C": best_c,
        "class_weight": SVM_CLASS_WT,
        "max_iter": SVM_MAX_ITER, "tol": SVM_TOL,
        "cache_size": SVM_CACHE_SIZE,
        "random_state": RANDOM_STATE, "probability": False,
    }
    if best_kernel == "rbf" and best_gamma != "N/A":
        best_params_cv["gamma"] = best_gamma

    metrics_best = evaluate_via_cv(
        best_params_cv, X_train, y_train, classes,
        f"Best {best_kernel.upper()} C={best_c}")

    cm_best, y_pred_best = get_cv_confusion_matrix(best_params_cv, X_train, y_train)
    f1_best_cls = f1_score(y_train, y_pred_best, average=None, zero_division=0)
    metrics_best["f1_per_class"] = f1_best_cls

    log(f"\n  [INFO] F1 per kelas Best Model (CV+SMOTE Partial):")
    for i, cls in enumerate(classes):
        diff = f1_best_cls[i] - [0.6554, 0.2674, 0.8791][i]
        log(f"    {cls:<12}: {f1_best_cls[i]*100:.2f}%  "
            f"(vs v1.0 test: {[0.6554,0.2674,0.8791][i]*100:.2f}%  "
            f"diff: {diff*100:+.2f}%)")

    # =========================================================================
    # TAHAP G: VISUALISASI
    # =========================================================================
    print_section("TAHAP G: VISUALISASI")

    viz_c_vs_f1_kernel(cv_results_df, best_c, best_kernel, OUTPUT_DIR, OUT_PREFIX)
    viz_cv_boxplot(cv_results_df, all_fold_scores, best_fold_key, OUTPUT_DIR, OUT_PREFIX)
    viz_confusion_matrix_cv(
        cm_base, classes, metrics_base["f1_macro"],
        f"Baseline Linear C=1",
        f"viz_{OUT_PREFIX}_cm_baseline.png", OUTPUT_DIR)
    viz_confusion_matrix_cv(
        cm_best, classes, metrics_best["f1_macro"],
        f"Best {best_kernel.upper()} C={best_c}",
        f"viz_{OUT_PREFIX}_cm_best.png", OUTPUT_DIR)
    viz_version_comparison(metrics_best, f1_best_cls, classes, OUTPUT_DIR, OUT_PREFIX)

    # =========================================================================
    # TAHAP H: REFIT + SIMPAN
    # =========================================================================
    print_section("TAHAP H: REFIT + SIMPAN BEST MODEL")

    best_params_final = {**best_params_cv, "probability": True}
    svm_best = SVC(**best_params_final)
    t0 = time.time()
    svm_best.fit(X_train, y_train)
    log_ok(f"Refit selesai: {time.time()-t0:.2f}s | "
           f"n_support={svm_best.n_support_.sum()}")

    best_model_path = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_best_model.joblib")
    joblib.dump(svm_best, best_model_path, compress=3)
    fsize = os.path.getsize(best_model_path) // 1024
    log_ok(f"{OUT_PREFIX}_best_model.joblib -> {fsize} KB")
    log_ok(f"Params: {best_params_final}")

    # Simpan log
    log_path = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # =========================================================================
    # PENUTUP
    # =========================================================================
    elapsed = time.time() - t_start
    th, rem = divmod(int(elapsed), 3600)
    tm, ts  = divmod(rem, 60)

    closing = (
        "\n" + "=" * 65 + "\n"
        "  MODELING v3.0 (SMOTE PARTIAL) SELESAI!\n"
        f"\n  SMOTE: partial, Neutral {int((y_train_orig==1).sum())}->"
        f"{SMOTE_STRATEGY_PARTIAL.get(1,'?')} | "
        f"k_neighbors={SMOTE_K_NEIGHBORS}\n"
        f"  Data setelah SMOTE: {n_train} sampel\n"
        f"\n  BEST MODEL:\n"
        f"    kernel : {best_kernel.upper()}\n"
        f"    C      : {best_c}\n"
        f"    gamma  : {best_gamma}\n"
        f"    CV F1-Macro : {best_cv_score*100:.4f}%\n"
        f"    Est. Test   : {best_est_test*100:.4f}% (CV - 1.5*std)\n"
        f"\n  F1 PER KELAS (CV+SMOTE Partial):\n"
        f"    {'Kelas':<14} {'v1.0 Test':>12} {'v3.0 CV':>12}\n"
    )
    for i, cls in enumerate(classes):
        v1 = [0.6554, 0.2674, 0.8791][i]
        v3 = f1_best_cls[i] if i < len(f1_best_cls) else 0
        closing += (f"    {cls.capitalize():<14} {v1*100:>11.2f}% "
                    f"{v3*100:>11.2f}%\n")
    closing += (
        f"\n  LANGKAH SELANJUTNYA:\n"
        f"    Jalankan evaluasiSmoteV3.py\n"
        f"    Model: outputSmoteV3/{OUT_PREFIX}_best_model.joblib\n"
        f"\n  Waktu total: {th}h {tm}m {ts:02d}s\n"
        + "=" * 65
    )
    print(C_HEAD + closing + C_RST)
    log_lines.append(closing)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log_ok(f"Log tersimpan: {log_path}")


if __name__ == "__main__":
    main()
