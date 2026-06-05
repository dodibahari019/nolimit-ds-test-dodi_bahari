"""
=============================================================================
TAHAP MODELING & HYPERPARAMETER TUNING: SVM — SBERT Embedding
=============================================================================
Proyek  : Sentiment Analysis Review (Indonesian)
Tahap   : Modeling + Hyperparameter Tuning — SVM + SBERT
Versi   : v1.0

=============================================================================
RUANG LINGKUP TAHAP INI
=============================================================================
  Tahap ini mencakup DUA sub-proses dalam satu file:
    [A] BASELINE MODELING  : Train SVC(kernel='linear', C=1.0), evaluasi via CV
    [B] HYPERPARAMETER TUNING: GridSearchCV — kernel LINEAR vs RBF + C + gamma
    OUTPUT AKHIR: best model yang siap dibawa ke tahap EVALUASI

  DATA YANG DIPAKAI DI TAHAP INI:
    data/X_train_emb.npy  + data/y_train.npy  -> TRAINING + CV (tuning)
    data/label_classes.npy                    -> Mapping label kelas

  DATA YANG TIDAK DISENTUH DI TAHAP INI:
    data/X_test_emb.npy  + data/y_test.npy   -> DISIMPAN untuk tahap EVALUASI
    !! Membocorkan X_test ke tuning = DATA LEAKAGE !!
    !! X_test HANYA dipakai di tahap evaluasi terpisah !!

  CATATAN PENTING:
    - Kasus ini: TRAIN + TEST saja (tidak ada split val terpisah)
    - Evaluasi di tahap ini menggunakan CV internal (StratifiedKFold)
    - SBERT embedding 384-dim, L2-normalized, float32

=============================================================================
KEPUTUSAN TEKNIS & VALIDASI
=============================================================================

[KEPUTUSAN 1] SVC untuk SBERT dense embedding
  ALASAN: SBERT menghasilkan dense float32 embedding 384-dim yang
  sudah L2-normalized. SVM (khususnya linear kernel) sangat efektif untuk
  data high-dimensional dense karena:
    - Embedding sudah terseparasi baik di ruang high-dim (inner product friendly)
    - Linear kernel mencari hyperplane optimal di ruang 384-dim tersebut
    - RBF kernel tetap dicoba untuk menangkap pola non-linear sentimen
  REF: Emergentmind (2026) — linear SVM pada SBERT embedding mencapai
       performa hampir setara BERT fine-tuning.

[KEPUTUSAN 2] GRID KERNEL: LINEAR + RBF — keduanya diuji
  Linear sering terbaik untuk SBERT karena SBERT sudah meng-encode
  semantic complexity dalam representasinya.
  RBF tetap dicoba: sentimen bahasa Indonesia bisa punya pola non-linear
  (sarkasme, ironi, campuran bahasa gaul).
  Argumen akademis lebih kuat dengan perbandingan keduanya.

[KEPUTUSAN 3] RANGE C: [0.1, 1, 10, 100]
  C=0.01 tidak digunakan karena data L2-normalized → underfitting parah.
  Range log-scale ini standar untuk SVM pada teks embedding.
  REF: Wainer & Fonseca (2021) — range C log-scale standar untuk SVM.
  REF: sklearn docs "Practical notes on SVM" — normalized data C >= 0.1.

[KEPUTUSAN 4] GAMMA GRID: ['scale', 'auto', 0.01, 0.001] — hanya untuk RBF
  gamma='scale' = 1/(n_features * X.var()) — adaptif, default sklearn
  gamma='auto'  = 1/n_features ≈ 1/384 ≈ 0.0026 — lebih konservatif
  gamma=0.01    — boundary smooth, generalize baik
  gamma=0.001   — boundary sangat smooth, mendekati linear
  gamma=0.1 TIDAK digunakan: terlalu besar untuk 384-dim normalized data,
  hampir pasti overfit pada dataset sentimen yang imbalanced.
  REF: TDS "SVM Hyperparameters Explained" — gamma besar = overfit.

[KEPUTUSAN 5] X_test TIDAK DISENTUH di tahap ini
  CV internal (StratifiedKFold) sudah cukup untuk memilih best params.
  X_test hanya dipakai sekali di tahap EVALUASI terpisah.
  REF: Hastie et al. (2009) — test set hanya untuk estimasi generalization
       error final, bukan untuk model selection.

[KEPUTUSAN 6] class_weight='balanced' — WAJIB
  Kelas neutral biasanya minoritas (~5%) dalam review sentimen.
  Balanced weighting mencegah model bias ke kelas mayoritas.

[KEPUTUSAN 7] probability=False untuk tuning (lebih cepat)
  Probability estimation via Platt scaling memperlambat training.
  Diaktifkan hanya di best model final untuk mendapatkan confidence score.

[KEPUTUSAN 8] Stratified 5-Fold CV — WAJIB imbalanced
  Memastikan proporsi kelas sama di setiap fold.
  Kritis untuk 3-class problem dengan kelas neutral yang minoritas.

[KEPUTUSAN 9] Scoring: F1-Macro — METRIK UTAMA
  Merata-ratakan kelas secara adil (negative, neutral, positive).
  Tidak terpengaruh imbalance seperti accuracy.

=============================================================================
INPUT (YANG DIPAKAI):
  data/X_train_emb.npy    -> (N_train x 384) float32, L2-normalized
  data/y_train.npy        -> Label train (int: 0=negative,1=neutral,2=positive)
  data/label_classes.npy  -> ['negative', 'neutral', 'positive']

INPUT (TIDAK DISENTUH — UNTUK TAHAP EVALUASI):
  data/X_test_emb.npy     -> JANGAN DILOAD DI SINI
  data/y_test.npy         -> JANGAN DILOAD DI SINI

OUTPUT TAHAP INI:
  output/svm_sbert_best_model.joblib         -> SVC best params (bawa ke Evaluasi)
  output/svm_sbert_baseline_model.joblib     -> SVC Linear C=1.0 baseline
  output/svm_sbert_tuning_results.csv        -> Semua kombinasi + CV scores
  output/viz_c_vs_f1_kernel.png              -> C vs F1-Macro per kernel (line chart)
  output/viz_heatmap_c_gamma.png             -> Heatmap C x gamma RBF (UNGGULAN)
  output/viz_kernel_comparison.png           -> Bar chart kernel Linear vs RBF
  output/viz_cv_boxplot.png                  -> Boxplot distribusi CV top kombinasi
  output/viz_confusion_matrix_baseline.png   -> CM baseline (dari CV)
  output/viz_confusion_matrix_best.png       -> CM best model (dari CV)
  output/viz_baseline_vs_best.png            -> Bar chart perbandingan semua metrik
  output/viz_class_f1_per_kernel.png         -> F1 per kelas (3 kelas) per kernel
  output/modeling_tuning_svm_sbert_log.txt   -> Log lengkap

=============================================================================
INSTALL:
  pip install scikit-learn numpy pandas joblib matplotlib colorama
=============================================================================
"""

import os
import time
import warnings
import random
import numpy as np
import pandas as pd
import joblib

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

warnings.filterwarnings("ignore")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(CURRENT_DIR, "data")
OUTPUT_DIR  = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# !! HANYA train yang di-load di tahap ini !!
IN_X_TRAIN = os.path.join(DATA_DIR, "X_train_emb.npy")
IN_Y_TRAIN = os.path.join(DATA_DIR, "y_train.npy")
IN_CLASSES = os.path.join(DATA_DIR, "label_classes.npy")
# X_test & y_test sengaja tidak didefinisikan — cegah data leakage

# =============================================================================
# PARAMETER MODELING & TUNING
# =============================================================================

# Baseline: SVC Linear C=1.0
BASELINE_C      = 1.0
BASELINE_KERNEL = "linear"

SVM_CLASS_WT    = "balanced"   # WAJIB untuk imbalanced 3-class
SVM_MAX_ITER    = 5000         # Cukup untuk SBERT 384-dim
SVM_TOL         = 1e-4
SVM_CACHE_SIZE  = 500          # MB — penting untuk RBF kernel
RANDOM_STATE    = 42

# Grid Search
# [KEPUTUSAN 3] C range log-scale, tidak ada 0.01
C_GRID = [0.1, 1, 10, 100]

# [KEPUTUSAN 2] Kedua kernel diuji
KERNEL_GRID = ["linear", "rbf"]

# [KEPUTUSAN 4] Gamma hanya untuk RBF, tanpa 0.1
GAMMA_GRID = ["scale", "auto", 0.01, 0.001]

# CV
CV_FOLDS     = 5
SCORING_MAIN = "f1_macro"

SCORING_MULTI = {
    "f1_macro"        : "f1_macro",
    "f1_weighted"     : "f1_weighted",
    "precision_macro" : "precision_macro",
    "recall_macro"    : "recall_macro",
    "accuracy"        : "accuracy",
}

# Visualisasi warna
VIZ_BG       = "#f8f9fa"
VIZ_GRID_CLR = "#dee2e6"
VIZ_LINEAR   = "#2980b9"
VIZ_RBF      = "#8e44ad"
VIZ_BASELINE = "#95a5a6"
VIZ_BEST     = "#e67e22"
VIZ_NEG      = "#e74c3c"
VIZ_NEU      = "#f39c12"
VIZ_POS      = "#27ae60"

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
# HELPER: EVALUASI VIA CV (bukan test set!)
# =============================================================================
def evaluate_via_cv(svc_params, X_train, y_train, classes, label):
    """
    Evaluasi model via cross_validate pada train set.
    TIDAK menggunakan X_test — tidak ada data leakage.
    """
    model = SVC(**svc_params)
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    cv_out = cross_validate(
        model, X_train, y_train,
        cv=skf, scoring=SCORING_MULTI,
        n_jobs=-1, return_train_score=False,
    )

    metrics = {
        "accuracy"             : cv_out["test_accuracy"].mean(),
        "precision_macro"      : cv_out["test_precision_macro"].mean(),
        "recall_macro"         : cv_out["test_recall_macro"].mean(),
        "f1_macro"             : cv_out["test_f1_macro"].mean(),
        "f1_weighted"          : cv_out["test_f1_weighted"].mean(),
        "f1_macro_std"         : cv_out["test_f1_macro"].std(),
        "f1_macro_per_fold"    : cv_out["test_f1_macro"].tolist(),
    }

    log_lines.append(f"\n[CV EVAL — {label}]")
    log_lines.append(f"  F1-Macro   : {metrics['f1_macro']*100:.4f}% "
                     f"(±{metrics['f1_macro_std']*100:.4f}%)  <- METRIK UTAMA")
    log_lines.append(f"  F1-Weighted: {metrics['f1_weighted']*100:.4f}%")
    log_lines.append(f"  Accuracy   : {metrics['accuracy']*100:.4f}%")
    log_lines.append(f"  Per fold   : "
                     f"{[f'{v*100:.2f}%' for v in metrics['f1_macro_per_fold']]}")
    return metrics


def get_cv_confusion_matrix(svc_params, X_train, y_train):
    """
    Confusion matrix dari cross_val_predict — agregat semua fold.
    Bukan dari X_test! Representatif untuk perilaku CV.
    """
    model  = SVC(**svc_params)
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_pred = cross_val_predict(model, X_train, y_train, cv=skf, n_jobs=-1)
    return confusion_matrix(y_train, y_pred), y_pred


# =============================================================================
# VISUALISASI 1 — C vs F1-Macro PER KERNEL (Line Chart)
# =============================================================================
def viz_c_vs_f1_kernel(cv_results_df, best_c, best_kernel, output_dir):
    """
    Line chart: C (log scale) vs mean CV F1-Macro, dipisah per kernel.
    Panel kiri : F1-Macro per kernel (metrik utama).
    Panel kanan: F1-Weighted per kernel (metrik pendukung).
    Shaded area = ±1 std.
    """
    df_lin = cv_results_df[cv_results_df["kernel"] == "linear"].sort_values("C")
    df_rbf = cv_results_df[cv_results_df["kernel"] == "rbf"].sort_values("C")

    df_lin_f1 = df_lin.groupby("C", as_index=False).agg(
        mean_f1=("mean_f1_macro", "max"), std_f1=("std_f1_macro", "min"))
    df_rbf_f1 = df_rbf.groupby("C", as_index=False).agg(
        mean_f1=("mean_f1_macro", "max"), std_f1=("std_f1_macro", "min"))

    df_lin_fw = df_lin.groupby("C", as_index=False).agg(
        mean_fw=("mean_f1_weighted", "max"))
    df_rbf_fw = df_rbf.groupby("C", as_index=False).agg(
        mean_fw=("mean_f1_weighted", "max"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    # ── Panel Kiri: F1-Macro ──────────────────────────────────────────────
    ax1.set_facecolor(VIZ_BG)
    if not df_lin_f1.empty:
        ax1.semilogx(df_lin_f1["C"], df_lin_f1["mean_f1"],
                     color=VIZ_LINEAR, marker="o", linewidth=2.2,
                     markersize=7, label="Linear Kernel")
        ax1.fill_between(df_lin_f1["C"],
                         df_lin_f1["mean_f1"] - df_lin_f1["std_f1"],
                         df_lin_f1["mean_f1"] + df_lin_f1["std_f1"],
                         alpha=0.12, color=VIZ_LINEAR)
    if not df_rbf_f1.empty:
        ax1.semilogx(df_rbf_f1["C"], df_rbf_f1["mean_f1"],
                     color=VIZ_RBF, marker="s", linewidth=2.2,
                     markersize=7, label="RBF Kernel")
        ax1.fill_between(df_rbf_f1["C"],
                         df_rbf_f1["mean_f1"] - df_rbf_f1["std_f1"],
                         df_rbf_f1["mean_f1"] + df_rbf_f1["std_f1"],
                         alpha=0.12, color=VIZ_RBF)

    best_row = cv_results_df.loc[cv_results_df["mean_f1_macro"].idxmax()]
    ax1.scatter([best_row["C"]], [best_row["mean_f1_macro"]],
                color=VIZ_BEST, s=220, zorder=6, marker="*",
                label=f"Best: {best_kernel.upper()} C={best_c}")
    ax1.axvline(x=best_c, color="#c0392b", linestyle=":", linewidth=2.0, alpha=0.8)

    ax1.set_xlabel("Nilai C (log scale)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Mean CV F1-Macro", fontsize=11, labelpad=8)
    ax1.set_title(
        f"C vs F1-Macro per Kernel — SVM SBERT Sentiment\n"
        f"Shaded = ±1 std | Best: {best_kernel.upper()} kernel, C={best_c}\n"
        f"CV F1-Macro = {best_row['mean_f1_macro']*100:.2f}%",
        fontsize=10, fontweight="bold", color="#212529", pad=10
    )
    ax1.legend(fontsize=9, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax1.grid(color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    # ── Panel Kanan: F1-Weighted ──────────────────────────────────────────
    ax2.set_facecolor(VIZ_BG)
    if not df_lin_fw.empty:
        ax2.semilogx(df_lin_fw["C"], df_lin_fw["mean_fw"],
                     color=VIZ_LINEAR, marker="o", linewidth=2.0,
                     markersize=6, label="Linear Kernel")
    if not df_rbf_fw.empty:
        ax2.semilogx(df_rbf_fw["C"], df_rbf_fw["mean_fw"],
                     color=VIZ_RBF, marker="s", linewidth=2.0,
                     markersize=6, label="RBF Kernel")
    ax2.axvline(x=best_c, color="#c0392b", linestyle=":",
                linewidth=2.0, alpha=0.8, label=f"Best C={best_c}")

    ax2.set_xlabel("Nilai C (log scale)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Mean CV F1-Weighted", fontsize=11, labelpad=8)
    ax2.set_title(
        "C vs F1-Weighted per Kernel\n"
        "F1-Weighted mempertimbangkan proporsi kelas\n"
        "Bandingkan dengan F1-Macro (panel kiri)",
        fontsize=10, fontweight="bold", color="#212529", pad=10
    )
    ax2.legend(fontsize=9, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax2.grid(color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    fig.suptitle(
        "Analisis Hyperparameter: C — SVM + SBERT (Sentiment Analysis)\n"
        f"Stratified {CV_FOLDS}-Fold CV pada Train Set | X_test TIDAK disentuh",
        fontsize=12, fontweight="bold", color="#212529", y=1.02
    )
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_c_vs_f1_kernel.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_c_vs_f1_kernel.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VISUALISASI 2 — HEATMAP C x GAMMA (RBF) — VISUALISASI UNGGULAN
# =============================================================================
def viz_heatmap_c_gamma(cv_results_df, best_c, best_gamma, output_dir):
    """
    Heatmap F1-Macro: baris=C, kolom=gamma, khusus kernel RBF.
    VISUALISASI UNGGULAN: interaksi C dan gamma langsung terlihat.
    Panel kiri : F1-Macro (metrik utama).
    Panel kanan: F1-Weighted (metrik pendukung).
    Kotak merah = kombinasi best model terpilih.
    """
    df_rbf = cv_results_df[cv_results_df["kernel"] == "rbf"].copy()
    if df_rbf.empty:
        log_warn("Tidak ada data RBF untuk heatmap. Skip.")
        return None

    df_rbf["gamma_str"] = df_rbf["gamma"].astype(str)
    gamma_order = [str(g) for g in GAMMA_GRID if str(g) in df_rbf["gamma_str"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    for ax_idx, (metric_col, metric_label, cmap_name) in enumerate([
        ("mean_f1_macro",    "F1-Macro (CV)",    "Blues"),
        ("mean_f1_weighted", "F1-Weighted (CV)", "Greens"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor(VIZ_BG)

        pivot_m = df_rbf.pivot_table(
            values=metric_col, index="C",
            columns="gamma_str", aggfunc="max")
        pivot_m = pivot_m.reindex(columns=gamma_order)

        vals = pivot_m.values.astype(float)
        vmin = np.nanmin(vals) - 0.005
        vmax = np.nanmax(vals) + 0.003

        im = ax.imshow(vals, cmap=cmap_name, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=f"{metric_label} Score")

        best_gamma_str = str(best_gamma)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isnan(v):
                    is_best = (
                        pivot_m.index[i] == best_c and
                        pivot_m.columns[j] == best_gamma_str and
                        ax_idx == 0
                    )
                    txt_color = "white" if (v - vmin) > (vmax - vmin) * 0.65 else "black"
                    ax.text(j, i, f"{v*100:.2f}%",
                            ha="center", va="center", fontsize=9.5,
                            fontweight="bold" if is_best else "normal",
                            color=txt_color)
                    if is_best:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="#c0392b", linewidth=2.5))

        ax.set_xticks(range(len(pivot_m.columns)))
        ax.set_xticklabels(pivot_m.columns, fontsize=10)
        ax.set_yticks(range(len(pivot_m.index)))
        ax.set_yticklabels([str(c) for c in pivot_m.index], fontsize=10)
        ax.set_xlabel("Gamma", fontsize=11, labelpad=8)
        ax.set_ylabel("C", fontsize=11, labelpad=8)
        ax.set_title(
            f"Heatmap {metric_label} — RBF Kernel\n"
            f"Kotak merah = Best: C={best_c}, gamma={best_gamma}",
            fontsize=10, fontweight="bold", color="#212529", pad=10
        )

    fig.suptitle(
        "Heatmap Hyperparameter RBF SVM: C × Gamma — Sentiment Analysis SBERT\n"
        "Kiri: F1-Macro (metrik utama)  |  Kanan: F1-Weighted\n"
        "Kotak merah = kombinasi terpilih | Semua skor dari CV Train Set",
        fontsize=12, fontweight="bold", color="#212529", y=1.02
    )
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_heatmap_c_gamma.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_heatmap_c_gamma.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VISUALISASI 3 — KERNEL COMPARISON BAR CHART
# =============================================================================
def viz_kernel_comparison(cv_results_df, best_c, best_kernel, output_dir):
    """
    Bar chart: perbandingan F1-Macro terbaik per kernel (Linear vs RBF).
    Menunjukkan dengan jelas kernel mana yang lebih unggul untuk kasus ini.
    """
    kernel_best = cv_results_df.groupby("kernel").agg(
        best_f1_macro    =("mean_f1_macro",    "max"),
        best_f1_weighted =("mean_f1_weighted", "max"),
        best_prec_macro  =("mean_prec_macro",  "max"),
        best_rec_macro   =("mean_rec_macro",   "max"),
        best_accuracy    =("mean_accuracy",    "max"),
    ).reset_index()

    metrics = ["F1-Macro", "F1-Weighted", "Precision\nMacro",
               "Recall\nMacro", "Accuracy"]
    keys    = ["best_f1_macro", "best_f1_weighted", "best_prec_macro",
               "best_rec_macro", "best_accuracy"]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    colors_k   = {"linear": VIZ_LINEAR, "rbf": VIZ_RBF}
    offset_map = {0: -w/2, 1: w/2}

    for ki, (_, krow) in enumerate(kernel_best.iterrows()):
        kern = krow["kernel"]
        vals = [float(krow[k]) for k in keys]
        off  = offset_map.get(ki, 0)
        bars = ax.bar(x + off, vals, w,
                      label=f"{kern.upper()} Kernel",
                      color=colors_k.get(kern, VIZ_BEST),
                      alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.007,
                    f"{val*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#212529")

    ax.set_title(
        f"Perbandingan Kernel: Linear vs RBF — SVM Sentiment SBERT\n"
        f"Best: {best_kernel.upper()} Kernel (C={best_c})\n"
        "Nilai = skor terbaik per kernel dari semua kombinasi C (CV Train Set)",
        fontsize=10, fontweight="bold", color="#212529", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Skor CV Mean (Train Set)", fontsize=11, labelpad=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=10, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "viz_kernel_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_kernel_comparison.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VISUALISASI 4 — CV BOXPLOT TOP KOMBINASI
# =============================================================================
def viz_cv_boxplot(cv_results_df, all_fold_scores, best_key, output_dir):
    """
    Boxplot distribusi F1-Macro per fold untuk top-8 kombinasi.
    Menunjukkan stabilitas/konsistensi model — bukan hanya mean.
    Kotak sempit = model stabil antar fold.
    """
    top_combos = cv_results_df.nlargest(8, "mean_f1_macro")

    data_boxes   = []
    combo_labels = []
    best_idx     = None

    for i, (_, row) in enumerate(top_combos.iterrows()):
        kern  = row["kernel"]
        c_val = row["C"]
        g_val = row.get("gamma", "N/A")
        key   = f"{kern}_{c_val}_{g_val}"
        if key in all_fold_scores:
            data_boxes.append(all_fold_scores[key])
            label = f"{kern.upper()}\nC={c_val}"
            if kern == "rbf":
                label += f"\nγ={g_val}"
            combo_labels.append(label)
            if key == best_key:
                best_idx = len(data_boxes) - 1

    if not data_boxes:
        log_warn("Tidak ada data fold untuk boxplot.")
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(data_boxes)*1.6), 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    bp = ax.boxplot(data_boxes, labels=combo_labels, patch_artist=True,
                    notch=False,
                    medianprops=dict(color="#c0392b", linewidth=2.0),
                    flierprops=dict(marker="o", color="#e74c3c",
                                    alpha=0.6, markersize=5),
                    whiskerprops=dict(linewidth=1.2, linestyle="--"),
                    capprops=dict(linewidth=1.2))

    for i, (patch, (_, row)) in enumerate(zip(bp["boxes"], top_combos.iterrows())):
        color = VIZ_LINEAR if row["kernel"] == "linear" else VIZ_RBF
        is_b  = (i == best_idx)
        patch.set_facecolor(VIZ_BEST if is_b else color)
        patch.set_alpha(0.65 if is_b else 0.50)

    for i, (_, row) in enumerate(top_combos.iterrows()):
        key_fold = f"{row['kernel']}_{row['C']}_{row.get('gamma', 'N/A')}"
        if key_fold == best_key and i < len(ax.get_xticklabels()):
            ax.get_xticklabels()[i].set_color("#c0392b")
            ax.get_xticklabels()[i].set_fontweight("bold")

    for i, scores in enumerate(data_boxes):
        mean_v = np.mean(scores)
        ax.text(i + 1, mean_v + 0.004, f"{mean_v*100:.2f}%",
                ha="center", va="bottom", fontsize=8.5,
                color="#212529", fontweight="bold")

    leg = [mpatches.Patch(color=VIZ_BEST,    label="Best Model"),
           mpatches.Patch(color=VIZ_LINEAR,  label="Linear Kernel"),
           mpatches.Patch(color=VIZ_RBF,     label="RBF Kernel")]
    ax.legend(handles=leg, fontsize=9, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax.set_xlabel("Kombinasi Kernel + C (+ Gamma untuk RBF)", fontsize=10, labelpad=8)
    ax.set_ylabel("F1-Macro per Fold (CV Train Set)", fontsize=11, labelpad=8)
    ax.set_title(
        f"Distribusi CV F1-Macro per Fold — Top 8 Kombinasi SVM\n"
        f"Stratified {CV_FOLDS}-Fold | Oranye = Best Model\n"
        "Kotak sempit = model lebih stabil antar fold",
        fontsize=10, fontweight="bold", color="#212529", pad=10
    )
    ax.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "viz_cv_boxplot.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_cv_boxplot.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VISUALISASI 5 — CONFUSION MATRIX (dari CV, bukan test set)
# =============================================================================
def viz_confusion_matrix_cv(cm, classes, cv_f1_macro, title_suffix,
                             filename, output_dir):
    """
    CM dari cross_val_predict — agregat semua fold pada Train Set.
    Bukan dari X_test! Menggambarkan perilaku model pada data unseen
    dalam konteks CV, tanpa bocorkan informasi test set.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest",
                   cmap=plt.cm.Blues, vmin=0, vmax=1)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_yticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_xlabel("Prediksi", fontsize=11, labelpad=6)
    ax.set_ylabel("Aktual", fontsize=11, labelpad=6)
    ax.set_title(
        f"Confusion Matrix (CV) — {title_suffix}\n"
        f"CV F1-Macro: {cv_f1_macro*100:.2f}% | 3 Kelas Sentimen\n"
        "Sumber: cross_val_predict Train Set (bukan test set!)",
        fontsize=11, fontweight="bold", color="#212529", pad=10
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"{filename} ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VISUALISASI 6 — BASELINE vs BEST (CV Score)
# =============================================================================
def viz_baseline_vs_best(metrics_base_cv, metrics_best_cv,
                          best_c, best_kernel, best_gamma, classes, output_dir):
    """
    Bar chart perbandingan CV score baseline vs best model.
    Panel kiri : semua metrik agregat.
    Panel kanan: F1 per kelas (3 kelas sentimen).
    Semua angka dari CV Train Set — tidak ada X_test.
    """
    metric_labels = ["Accuracy", "F1-Macro", "F1-Weighted",
                     "Precision\nMacro", "Recall\nMacro"]
    keys          = ["accuracy", "f1_macro", "f1_weighted",
                     "precision_macro", "recall_macro"]

    base_vals = [metrics_base_cv[k] for k in keys]
    best_vals = [metrics_best_cv[k] for k in keys]
    x = np.arange(len(metric_labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    color_best = VIZ_LINEAR if best_kernel == "linear" else VIZ_RBF

    # Panel kiri: bar semua metrik
    ax1.set_facecolor(VIZ_BG)
    b1 = ax1.bar(x - w/2, base_vals, w,
                 label=f"Baseline (Linear C={BASELINE_C})",
                 color=VIZ_BASELINE, alpha=0.85, edgecolor="white")
    b2 = ax1.bar(x + w/2, best_vals, w,
                 label=f"Best ({best_kernel.upper()} C={best_c})",
                 color=color_best, alpha=0.85, edgecolor="white")

    for bar, val in zip(list(b1) + list(b2), base_vals + best_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                 f"{val*100:.1f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")

    for i, (bv, bst) in enumerate(zip(base_vals, best_vals)):
        gain = bst - bv
        if abs(gain) > 0.001:
            color_g = "#27ae60" if gain > 0 else "#e74c3c"
            ax1.text(x[i] + w/2, bst + 0.032, f"{gain*100:+.1f}%",
                     ha="center", va="bottom", fontsize=7.5,
                     color=color_g, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=10)
    ax1.set_ylabel("Skor CV Mean (Train Set)", fontsize=11, labelpad=8)
    ax1.set_ylim(0, 1.18)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax1.set_title(
        f"Baseline vs Best Model — Semua Metrik (CV)\n"
        f"Best: {best_kernel.upper()} C={best_c}"
        f"{' γ='+str(best_gamma) if best_kernel=='rbf' else ''}\n"
        "!! Skor dari CV Train Set — X_test BELUM disentuh !!",
        fontsize=10, fontweight="bold", color="#212529", pad=10
    )
    ax1.legend(fontsize=9, framealpha=0.85, edgecolor=VIZ_GRID_CLR)
    ax1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel kanan: F1 per kelas (3 kelas sentimen)
    ax2.set_facecolor(VIZ_BG)
    cls_colors = [VIZ_NEG, VIZ_NEU, VIZ_POS]
    n_cls = len(classes)
    x2    = np.arange(n_cls)

    f1_base_cls = metrics_base_cv.get("f1_per_class", np.zeros(n_cls))[:n_cls]
    f1_best_cls = metrics_best_cv.get("f1_per_class", np.zeros(n_cls))[:n_cls]

    b3 = ax2.bar(x2 - w/2, f1_base_cls, w, label="Baseline",
                 color=VIZ_BASELINE, alpha=0.80, edgecolor="white")
    b4 = ax2.bar(x2 + w/2, f1_best_cls, w, label="Best Model",
                 color=cls_colors[:n_cls], alpha=0.80, edgecolor="white")

    for bar, val in zip(list(b3) + list(b4),
                        list(f1_base_cls) + list(f1_best_cls)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                 f"{val*100:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
    ax2.set_ylabel("CV F1-Score per Kelas", fontsize=11, labelpad=8)
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax2.set_title(
        "CV F1-Score per Kelas Sentimen\n"
        "Neutral biasanya terendah (kelas minoritas)\n"
        "Warna: Merah=Negative | Kuning=Neutral | Hijau=Positive",
        fontsize=10, fontweight="bold", color="#212529", pad=10
    )
    ax2.legend(fontsize=10, framealpha=0.85)
    ax2.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Perbandingan Baseline vs Best Model — SVM + SBERT Sentiment\n"
        f"Semua skor dari CV Train Set | X_test TIDAK disentuh (untuk tahap Evaluasi)",
        fontsize=12, fontweight="bold", color="#212529", y=1.02
    )
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_baseline_vs_best.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_baseline_vs_best.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VISUALISASI 7 — F1 PER KELAS DETAIL PER KERNEL
# =============================================================================
def viz_class_f1_per_kernel(y_train, y_pred_base, y_pred_best,
                             classes, best_kernel, best_c, output_dir):
    """
    Perbandingan F1, Precision, Recall per kelas sentimen.
    Sumber: cross_val_predict pada Train Set (bukan X_test).
    3 panel: 1 per kelas (negative, neutral, positive).
    Menunjukkan kekuatan/kelemahan model per kelas secara detail.
    """
    n_cls      = len(classes)
    cls_colors = [VIZ_NEG, VIZ_NEU, VIZ_POS]

    f1_base  = f1_score(y_train, y_pred_base,        average=None, zero_division=0)
    p_base   = precision_score(y_train, y_pred_base, average=None, zero_division=0)
    r_base   = recall_score(y_train, y_pred_base,    average=None, zero_division=0)
    f1_best  = f1_score(y_train, y_pred_best,        average=None, zero_division=0)
    p_best   = precision_score(y_train, y_pred_best, average=None, zero_division=0)
    r_best   = recall_score(y_train, y_pred_best,    average=None, zero_division=0)

    fig, axes = plt.subplots(1, n_cls, figsize=(14, 5))
    fig.patch.set_facecolor(VIZ_BG)

    for i, (cls_name, color) in enumerate(zip(classes, cls_colors)):
        ax = axes[i]
        ax.set_facecolor(VIZ_BG)

        bv_f1 = f1_base[i] if i < len(f1_base) else 0
        bv_p  = p_base[i]  if i < len(p_base)  else 0
        bv_r  = r_base[i]  if i < len(r_base)  else 0
        bst_f1 = f1_best[i] if i < len(f1_best) else 0
        bst_p  = p_best[i]  if i < len(p_best)  else 0
        bst_r  = r_best[i]  if i < len(r_best)  else 0

        metrics_list = ["F1-Score", "Precision", "Recall"]
        base_vals    = [bv_f1, bv_p, bv_r]
        best_vals    = [bst_f1, bst_p, bst_r]
        y_pos = np.arange(len(metrics_list))

        ax.barh(y_pos + 0.2, best_vals, height=0.35, color=color,
                alpha=0.85, label="Best Model", edgecolor="white")
        ax.barh(y_pos - 0.2, base_vals, height=0.35, color=VIZ_BASELINE,
                alpha=0.70, label=f"Baseline (Linear C={BASELINE_C})", edgecolor="white")

        for j, (bv, bst) in enumerate(zip(base_vals, best_vals)):
            ax.text(bst + 0.01, j + 0.2, f"{bst*100:.1f}%",
                    va="center", fontsize=8.5, fontweight="bold")
            ax.text(bv + 0.01, j - 0.2, f"{bv*100:.1f}%",
                    va="center", fontsize=8.5, color="#666666")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics_list, fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_title(
            f"Kelas: {cls_name.upper()}\n"
            f"Best F1 (CV): {bst_f1*100:.1f}%",
            fontsize=11, fontweight="bold", color=color, pad=8
        )
        ax.grid(axis="x", color=VIZ_GRID_CLR, linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(fontsize=7.5, framealpha=0.85, loc="lower right")

    fig.suptitle(
        f"F1 / Precision / Recall per Kelas Sentimen (CV Train Set)\n"
        f"Baseline vs Best Model ({best_kernel.upper()} C={best_c})\n"
        "Sumber: cross_val_predict | X_test TIDAK disentuh",
        fontsize=12, fontweight="bold", color="#212529", y=1.02
    )
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_class_f1_per_kernel.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_class_f1_per_kernel.png ({os.path.getsize(path)//1024} KB)")
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
        "  MODELING & HYPERPARAMETER TUNING — SVM + SBERT\n"
        "  Proyek : Sentiment Analysis Review (Indonesian)\n"
        f"  Start  : {timestamp}\n"
        "  INPUT DIPAKAI  : data/X_train_emb.npy + data/y_train.npy\n"
        "  INPUT DISIMPAN : data/X_test_emb.npy  + data/y_test.npy\n"
        "                   -> X_test TIDAK DISENTUH di tahap ini!\n"
        "  ALGORITMA : SVC (C-Support Vector Classification)\n"
        f"  GRID      : kernel={KERNEL_GRID}\n"
        f"              C={C_GRID}\n"
        f"              gamma={GAMMA_GRID} (RBF saja)\n"
        f"  CV        : StratifiedKFold({CV_FOLDS}) | scoring=f1_macro\n"
        f"  class_weight='balanced' — wajib untuk 3-class imbalanced\n"
        + "=" * 65
    )
    print(C_HEAD + banner + C_RST)
    log_lines.append(banner)

    # =========================================================================
    # TAHAP A: LOAD DATA (hanya train!)
    # =========================================================================
    print_section("TAHAP A: LOAD DATA (Train Only — Test Tidak Disentuh)")

    for path, name in [
        (IN_X_TRAIN, "X_train_emb.npy"),
        (IN_Y_TRAIN, "y_train.npy"),
        (IN_CLASSES, "label_classes.npy"),
    ]:
        if not os.path.exists(path):
            log_err(f"File tidak ditemukan: {path}")
            log_err("Pastikan feature_extraction_sbert.py sudah dijalankan.")
            return

    X_train = np.load(IN_X_TRAIN).astype(np.float32)
    y_train = np.load(IN_Y_TRAIN)
    classes = np.load(IN_CLASSES, allow_pickle=True).tolist()

    n_train    = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes  = len(classes)

    log_ok(f"X_train : {X_train.shape}  dtype={X_train.dtype}")
    log_ok(f"y_train : {y_train.shape}")
    log_ok(f"Classes : {classes}  ({n_classes} kelas)")
    log(f"\n  [PENTING] X_test & y_test TIDAK di-load di tahap ini.")
    log(f"  [PENTING] X_test disimpan untuk tahap Evaluasi terpisah.\n")

    # Distribusi kelas
    log_lines.append("\n[INFO] Distribusi kelas (Train):")
    for i, cls in enumerate(classes):
        cnt = int((y_train == i).sum())
        log(f"  {i} -> {cls:<12} | {cnt:>5} sampel ({cnt/n_train*100:.1f}%)")

    # Verifikasi L2-norm
    norms = np.linalg.norm(X_train[:10], axis=1)
    log_lines.append(f"\n[VERIFY] L2 norms sample (harus ~1.0): {norms.round(4).tolist()}")
    log_lines.append("[INFO] SBERT 384-dim L2-normalized -> tidak perlu StandardScaler tambahan")

    # =========================================================================
    # TAHAP B: BASELINE MODEL — SVC Linear C=1.0, evaluasi via CV
    # =========================================================================
    print_section("TAHAP B: BASELINE MODEL (SVC Linear C=1.0, Evaluasi via CV)")

    baseline_params = {
        "kernel"       : BASELINE_KERNEL,
        "C"            : BASELINE_C,
        "class_weight" : SVM_CLASS_WT,
        "max_iter"     : SVM_MAX_ITER,
        "tol"          : SVM_TOL,
        "cache_size"   : SVM_CACHE_SIZE,
        "random_state" : RANDOM_STATE,
        "probability"  : False,  # False dulu untuk kecepatan CV
    }
    log_lines.append(f"[INFO] Params baseline: {baseline_params}")
    log_lines.append("[INFO] Evaluasi via cross_validate — BUKAN X_test!")
    log_lines.append("[INFO] REF: Emergentmind (2026) — C=1 sering optimal untuk SBERT+SVM")

    metrics_base_cv = evaluate_via_cv(
        baseline_params, X_train, y_train, classes,
        f"Baseline SVC (Linear C={BASELINE_C})")

    # CM baseline dari CV (cross_val_predict)
    cm_base, y_pred_cv_base = get_cv_confusion_matrix(
        baseline_params, X_train, y_train)

    # F1 per kelas dari CV predictions
    f1_base_cls = f1_score(y_train, y_pred_cv_base, average=None, zero_division=0)
    metrics_base_cv["f1_per_class"] = f1_base_cls

    # Latih baseline pada seluruh train untuk disimpan
    svm_baseline_full = SVC(**{**baseline_params, "probability": True})
    t0 = time.time()
    svm_baseline_full.fit(X_train, y_train)
    log_ok(f"Baseline fit selesai dalam {time.time()-t0:.2f}s | "
           f"n_support={svm_baseline_full.n_support_.sum()}")
    log_ok(f"CV F1-Macro (baseline): {metrics_base_cv['f1_macro']*100:.4f}% "
           f"(±{metrics_base_cv['f1_macro_std']*100:.4f}%)")

    joblib.dump(svm_baseline_full,
                os.path.join(OUTPUT_DIR, "svm_sbert_baseline_model.joblib"),
                compress=3)
    log_ok("svm_sbert_baseline_model.joblib tersimpan")

    # =========================================================================
    # TAHAP C: HYPERPARAMETER TUNING — GridSearchCV
    # =========================================================================
    print_section("TAHAP C: HYPERPARAMETER TUNING — GridSearchCV")
    log(f"\n  [INFO] Grid: C={C_GRID} | kernel={KERNEL_GRID} | gamma={GAMMA_GRID}")
    log(f"  [INFO] gamma hanya untuk kernel='rbf'")
    log(f"  [INFO] Total kombinasi: "
        f"{len(C_GRID)} linear + {len(C_GRID)*len(GAMMA_GRID)} rbf = "
        f"{len(C_GRID) + len(C_GRID)*len(GAMMA_GRID)}")
    log_lines.append("[INFO] probability=False saat tuning — lebih cepat")
    log_lines.append("[INFO] X_test TIDAK masuk GridSearchCV sama sekali")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # Grid kondisional: gamma hanya untuk rbf
    param_grid = [
        {"C": C_GRID, "kernel": ["linear"]},
        {"C": C_GRID, "kernel": ["rbf"], "gamma": GAMMA_GRID},
    ]

    svm_for_grid = SVC(
        class_weight = SVM_CLASS_WT,
        probability  = False,      # False untuk kecepatan tuning
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

    t0 = time.time()
    log(f"\n  Menjalankan GridSearchCV... (bisa memakan beberapa menit untuk RBF)")
    grid_search.fit(X_train, y_train)
    dur_grid = time.time() - t0

    best_params   = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_c        = best_params["C"]
    best_kernel   = best_params["kernel"]
    best_gamma    = best_params.get("gamma", "N/A")

    log_ok(f"GridSearchCV selesai dalam {dur_grid:.2f}s")
    log_ok(f"BEST PARAMS : {best_params}")
    log_ok(f"BEST CV F1-Macro: {best_cv_score*100:.4f}%")

    # ── Kumpulkan hasil CV ─────────────────────────────────────────────────
    cv_res  = grid_search.cv_results_
    cv_rows = []
    all_fold_scores = {}

    for idx in range(len(cv_res["params"])):
        p     = cv_res["params"][idx]
        kern  = p.get("kernel", "?")
        c_val = p.get("C", "?")
        g_val = p.get("gamma", "N/A")

        mean_f1m = float(cv_res["mean_test_f1_macro"][idx])
        std_f1m  = float(cv_res["std_test_f1_macro"][idx])
        mean_f1w = float(cv_res.get("mean_test_f1_weighted",
                                    [0]*len(cv_res["params"]))[idx])
        mean_pm  = float(cv_res["mean_test_precision_macro"][idx])
        mean_rm  = float(cv_res["mean_test_recall_macro"][idx])
        mean_acc = float(cv_res["mean_test_accuracy"][idx])

        cv_rows.append({
            "kernel"         : kern,
            "C"              : c_val,
            "gamma"          : g_val,
            "mean_f1_macro"  : mean_f1m,
            "std_f1_macro"   : std_f1m,
            "mean_f1_weighted": mean_f1w,
            "mean_prec_macro": mean_pm,
            "mean_rec_macro" : mean_rm,
            "mean_accuracy"  : mean_acc,
        })

        fold_scores = [float(cv_res[f"split{k}_test_f1_macro"][idx])
                       for k in range(CV_FOLDS)]
        fold_key = f"{kern}_{c_val}_{g_val}"
        all_fold_scores[fold_key] = fold_scores

        best_flag = " <-- BEST" if (
            kern == best_kernel and c_val == best_c and
            str(g_val) == str(best_gamma)
        ) else ""
        log_lines.append(
            f"  {kern.upper()} C={c_val} gamma={g_val}: "
            f"F1={mean_f1m*100:.3f}% (±{std_f1m*100:.3f}%){best_flag}")

    cv_results_df = pd.DataFrame(cv_rows)
    best_fold_key = f"{best_kernel}_{best_c}_{best_gamma}"

    csv_path = os.path.join(OUTPUT_DIR, "svm_sbert_tuning_results.csv")
    cv_results_df.to_csv(csv_path, index=False)
    log_ok(f"svm_sbert_tuning_results.csv tersimpan ({len(cv_results_df)} kombinasi)")

    # =========================================================================
    # TAHAP D: EVALUASI BEST MODEL via CV (bukan test set!)
    # =========================================================================
    print_section("TAHAP D: EVALUASI BEST MODEL via CV (Bukan Test Set!)")
    log(f"\n  [PENTING] Evaluasi CV pada Train Set — X_test MASIH TERSIMPAN.\n")

    best_params_for_cv = {
        "kernel"       : best_kernel,
        "C"            : best_c,
        "class_weight" : SVM_CLASS_WT,
        "max_iter"     : SVM_MAX_ITER,
        "tol"          : SVM_TOL,
        "cache_size"   : SVM_CACHE_SIZE,
        "random_state" : RANDOM_STATE,
        "probability"  : False,
    }
    if best_kernel == "rbf" and best_gamma != "N/A":
        best_params_for_cv["gamma"] = best_gamma

    metrics_best_cv = evaluate_via_cv(
        best_params_for_cv, X_train, y_train, classes,
        f"Best SVC ({best_kernel.upper()} C={best_c})")

    # CM best dari CV
    cm_best, y_pred_cv_best = get_cv_confusion_matrix(
        best_params_for_cv, X_train, y_train)

    # F1 per kelas best
    f1_best_cls = f1_score(y_train, y_pred_cv_best, average=None, zero_division=0)
    metrics_best_cv["f1_per_class"] = f1_best_cls

    # Ringkasan gain
    log_lines.append("\n[RINGKASAN GAIN DARI TUNING — CV TRAIN SET]")
    log_lines.append(f"  {'Metrik':<22} {'Baseline CV':>14} {'Best CV':>14} {'Gain':>10}")
    for nm, key in [
        ("F1-Macro",       "f1_macro"),
        ("F1-Weighted",    "f1_weighted"),
        ("Accuracy",       "accuracy"),
        ("Prec (Macro)",   "precision_macro"),
        ("Recall (Macro)", "recall_macro"),
    ]:
        bv   = metrics_base_cv[key]
        bst  = metrics_best_cv[key]
        gain = bst - bv
        flag = " [GAIN!]"  if gain > 0.001 else \
               " [SAMA]"   if abs(gain) <= 0.001 else " [TURUN]"
        log_lines.append(
            f"  {nm:<22} {bv*100:>13.4f}% {bst*100:>13.4f}% "
            f"{gain*100:>+9.4f}%{flag}")

    # =========================================================================
    # TAHAP E: REFIT BEST MODEL PADA SELURUH TRAIN — dengan probability=True
    # =========================================================================
    print_section("TAHAP E: REFIT BEST MODEL (probability=True untuk Deployment)")
    log_lines.append("[INFO] probability=True diaktifkan agar bisa predict_proba()")
    log_lines.append("[INFO] Refit pada SELURUH X_train — bukan sebagian")

    best_params_final = {**best_params_for_cv, "probability": True}
    svm_best = SVC(**best_params_final)
    t0 = time.time()
    svm_best.fit(X_train, y_train)
    log_ok(f"Refit selesai dalam {time.time()-t0:.2f}s | "
           f"n_support={svm_best.n_support_.sum()}")

    # =========================================================================
    # TAHAP F: VISUALISASI
    # =========================================================================
    print_section("TAHAP F: MEMBUAT VISUALISASI")

    viz_c_vs_f1_kernel(cv_results_df, best_c, best_kernel, OUTPUT_DIR)
    viz_heatmap_c_gamma(cv_results_df, best_c, best_gamma, OUTPUT_DIR)
    viz_kernel_comparison(cv_results_df, best_c, best_kernel, OUTPUT_DIR)
    viz_cv_boxplot(cv_results_df, all_fold_scores, best_fold_key, OUTPUT_DIR)

    viz_confusion_matrix_cv(
        cm_base, classes, metrics_base_cv["f1_macro"],
        f"Baseline SVC (Linear C={BASELINE_C})",
        "viz_confusion_matrix_baseline.png", OUTPUT_DIR)

    viz_confusion_matrix_cv(
        cm_best, classes, metrics_best_cv["f1_macro"],
        f"Best SVC ({best_kernel.upper()} C={best_c})",
        "viz_confusion_matrix_best.png", OUTPUT_DIR)

    viz_baseline_vs_best(
        metrics_base_cv, metrics_best_cv,
        best_c, best_kernel, best_gamma, classes, OUTPUT_DIR)

    viz_class_f1_per_kernel(
        y_train, y_pred_cv_base, y_pred_cv_best,
        classes, best_kernel, best_c, OUTPUT_DIR)

    # =========================================================================
    # TAHAP G: SIMPAN BEST MODEL
    # =========================================================================
    print_section("TAHAP G: SIMPAN BEST MODEL")

    best_model_path = os.path.join(OUTPUT_DIR, "svm_sbert_best_model.joblib")
    joblib.dump(svm_best, best_model_path, compress=3)
    kb = os.path.getsize(best_model_path) / 1024
    log_ok(f"svm_sbert_best_model.joblib -> {kb:.1f} KB")
    log_ok(f"Params: {best_params_final}")

    # =========================================================================
    # TAHAP H: BANNER PENUTUP + LOG
    # =========================================================================
    elapsed = time.time() - t_start
    th, rem = divmod(int(elapsed), 3600)
    tm, ts  = divmod(rem, 60)

    closing = (
        "\n" + "=" * 65 + "\n"
        "  MODELING & TUNING SVM SBERT — SELESAI!\n"
        f"\n  DATA TRAIN:\n"
        f"    X_train : {X_train.shape} | SBERT 384-dim L2-normalized\n"
        f"    Kelas   : {classes}\n"
        f"\n  DATA TEST (TIDAK DISENTUH):\n"
        f"    X_test & y_test disimpan untuk tahap Evaluasi\n"
        f"\n  BEST MODEL:\n"
        f"    kernel  : {best_kernel.upper()}\n"
        f"    C       : {best_c}\n"
        f"    gamma   : {best_gamma if best_kernel == 'rbf' else 'N/A (linear)'}\n"
        f"    CV F1-Macro: {best_cv_score*100:.4f}% "
        f"(Stratified {CV_FOLDS}-Fold)\n"
        f"\n  RINGKASAN CV (Train Set — bukan test set):\n"
        f"    {'Metrik':<20} {'Baseline':>14} {'Best SVM':>14}\n"
        f"    {'F1-Macro':<20} "
        f"{metrics_base_cv['f1_macro']*100:>13.2f}% "
        f"{metrics_best_cv['f1_macro']*100:>13.2f}%\n"
        f"    {'F1-Weighted':<20} "
        f"{metrics_base_cv['f1_weighted']*100:>13.2f}% "
        f"{metrics_best_cv['f1_weighted']*100:>13.2f}%\n"
        f"    {'Accuracy':<20} "
        f"{metrics_base_cv['accuracy']*100:>13.2f}% "
        f"{metrics_best_cv['accuracy']*100:>13.2f}%\n"
        f"\n  OUTPUT (folder: output/):\n"
        f"    svm_sbert_best_model.joblib        <- BAWA ke tahap Evaluasi\n"
        f"    svm_sbert_baseline_model.joblib\n"
        f"    svm_sbert_tuning_results.csv\n"
        f"    viz_c_vs_f1_kernel.png\n"
        f"    viz_heatmap_c_gamma.png            <- UNGGULAN\n"
        f"    viz_kernel_comparison.png\n"
        f"    viz_cv_boxplot.png\n"
        f"    viz_confusion_matrix_baseline.png\n"
        f"    viz_confusion_matrix_best.png\n"
        f"    viz_baseline_vs_best.png\n"
        f"    viz_class_f1_per_kernel.png\n"
        f"    modeling_tuning_svm_sbert_log.txt\n"
        f"\n  CARA LOAD DI TAHAP EVALUASI:\n"
        f"    import joblib, numpy as np\n"
        f"    model  = joblib.load('output/svm_sbert_best_model.joblib')\n"
        f"    X_test = np.load('data/X_test_emb.npy')\n"
        f"    y_test = np.load('data/y_test.npy')\n"
        f"    # Evaluasi FINAL di tahap Evaluasi — bukan di sini!\n"
        f"\n  Waktu total: {th}h {tm}m {ts:02d}s\n"
        + "=" * 65
    )
    print(C_HEAD + closing + C_RST)
    log_lines.append(closing)

    log_path = os.path.join(OUTPUT_DIR, "modeling_tuning_svm_sbert_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log_ok(f"Log tersimpan: {log_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
