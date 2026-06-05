"""
=============================================================================
TAHAP EVALUASI FINAL: SVM + SBERT + SMOTE PARTIAL (v3.0)
=============================================================================
Proyek  : Sentiment Analysis Review (Indonesian)
Tahap   : Evaluasi Final Model v3.0
Versi   : v3.0 — SMOTE Partial, bukan full balance

=============================================================================
PERUBAHAN DARI evaluasiSmote.py (v2.0)
=============================================================================
  - Membaca model dari outputSmoteV3/ (bukan outputSmote/)
  - Label versi disesuaikan ke v3.0
  - Tambah perbandingan 3 versi: v1.0 / v2.0 / v3.0
  - Tambah analisis apakah SMOTE partial berhasil memperbaiki Neutral
  - Rekomendasi next step lebih spesifik

INPUT:
  outputSmoteV3/svm_sbert_smote_v3_best_model.joblib
  data/X_test_emb.npy   <- test set ORIGINAL, tidak di-SMOTE
  data/y_test.npy
  data/label_classes.npy

OUTPUT:
  outputSmoteV3/v3_eval_classification_report.txt
  outputSmoteV3/v3_eval_metrics_summary.csv
  outputSmoteV3/viz_v3_eval_confusion_matrix.png
  outputSmoteV3/viz_v3_eval_per_class_f1.png
  outputSmoteV3/viz_v3_eval_3versions_comparison.png  <- UNGGULAN
  outputSmoteV3/viz_v3_eval_roc_curve.png
  outputSmoteV3/viz_v3_eval_pr_curve.png
  outputSmoteV3/viz_v3_eval_confidence_dist.png
  outputSmoteV3/viz_v3_eval_summary_dashboard.png     <- UNGGULAN
  outputSmoteV3/v3_evaluation_log.txt
=============================================================================
"""

import os
import sys
import io
import time
import warnings
import numpy as np
import pandas as pd
import joblib

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# =============================================================================
# PATH
# =============================================================================
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(CURRENT_DIR, "data")
OUTPUT_DIR    = os.path.join(CURRENT_DIR, "outputSmote")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IN_BEST_MODEL = os.path.join(OUTPUT_DIR, "svm_sbert_smote_v3_best_model.joblib")
IN_X_TEST     = os.path.join(DATA_DIR, "X_test_emb.npy")
IN_Y_TEST     = os.path.join(DATA_DIR, "y_test.npy")
IN_CLASSES    = os.path.join(DATA_DIR, "label_classes.npy")

# =============================================================================
# REFERENSI HISTORIS (hardcoded dari evaluasi sebelumnya)
# =============================================================================
# v1.0: SVM Linear C=1, tanpa SMOTE
V1 = {
    "f1_macro"    : 0.6006, "f1_weighted": 0.8045, "accuracy"  : 0.7857,
    "f1_neg"      : 0.6554, "f1_neu"     : 0.2674, "f1_pos"    : 0.8791,
    "label"       : "v1.0\nLinear C=1\n(no SMOTE)",
}
# v2.0: SVM RBF C=100, SMOTE full x14 (GAGAL overfitting)
V2 = {
    "f1_macro"    : 0.5370, "f1_weighted": 0.8133, "accuracy"  : 0.8249,
    "f1_neg"      : 0.6638, "f1_neu"     : 0.0426, "f1_pos"    : 0.9047,
    "label"       : "v2.0\nRBF C=100\n(SMOTE full x14 - GAGAL)",
}

# =============================================================================
# WARNA
# =============================================================================
VIZ_BG       = "#f8f9fa"
VIZ_GRID_CLR = "#dee2e6"
VIZ_NEG      = "#e74c3c"
VIZ_NEU      = "#f39c12"
VIZ_POS      = "#27ae60"
VIZ_V1       = "#bdc3c7"
VIZ_V2       = "#e74c3c"
VIZ_V3       = "#16a085"
VIZ_V3_EST   = "#e67e22"
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
# VIZ 1 — CONFUSION MATRIX FINAL
# =============================================================================
def viz_confusion_matrix(cm, classes, metrics, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    n = len(classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    for ax, data, is_norm in [(ax1, cm, False), (ax2, cm_norm, True)]:
        ax.set_facecolor(VIZ_BG)
        im = ax.imshow(data, cmap="Blues",
                       vmin=0, vmax=(1 if is_norm else cm.max()))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                thr = (0.55 if is_norm else cm.max()*0.55)
                txt_c = "white" if val > thr else "black"
                if is_norm:
                    label = f"{cm[i,j]}\n({val*100:.1f}%)"
                else:
                    label = f"{int(val)}\n({cm_norm[i,j]*100:.1f}%)"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_c)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
        ax.set_yticklabels([c.capitalize() for c in classes], fontsize=10)
        ax.set_xlabel("Prediksi", fontsize=11)
        ax.set_ylabel("Aktual", fontsize=11)
        ax.set_title(
            f"CM {'Normalized' if is_norm else 'Raw'} — v3.0 (SMOTE Partial)\n"
            f"Test Set Original | F1-Macro: {metrics['f1_macro']*100:.2f}%",
            fontsize=10, fontweight="bold", pad=10)

    fig.suptitle(
        f"Confusion Matrix FINAL v3.0 — Test Set: {cm.sum()} sampel ORIGINAL\n"
        f"F1-Macro={metrics['f1_macro']*100:.2f}% | "
        f"Accuracy={metrics['accuracy']*100:.2f}% | "
        f"F1-Weighted={metrics['f1_weighted']*100:.2f}%",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_v3_eval_confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_v3_eval_confusion_matrix.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 2 — F1 PER KELAS
# =============================================================================
def viz_per_class_f1(f1s, prcs, recs, classes, supports, output_dir):
    n = len(classes)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    for mi, (vals, label, offset, hatch) in enumerate([
        (f1s,  "F1-Score",  -w,  ""),
        (prcs, "Precision",   0,  "//"),
        (recs, "Recall",     +w,  ".."),
    ]):
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=CLASS_COLORS[:n], alpha=0.80,
                      edgecolor="white", hatch=hatch if hatch else "")
        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                    f"{val*100:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Garis referensi v1.0
    v1_f1 = [V1["f1_neg"], V1["f1_neu"], V1["f1_pos"]]
    for i, v1 in enumerate(v1_f1):
        ax.hlines(v1, x[i]-w*1.6, x[i]+w*1.6,
                  colors=VIZ_V1, linewidth=2, linestyle="--",
                  label="_nolegend_")
    ax.plot([], [], color=VIZ_V1, linewidth=2, linestyle="--",
            label=f"v1.0 F1 benchmark")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c.capitalize()}\n(n={s})" for c, s in zip(classes, supports)],
        fontsize=11)
    ax.set_ylabel("Skor (Test Set Original)", fontsize=11)
    ax.set_ylim(0, 1.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    ax.set_title(
        "F1 / Precision / Recall per Kelas — v3.0 (SMOTE Partial)\n"
        f"Garis abu = v1.0 benchmark | Target: Neutral > {V1['f1_neu']*100:.1f}%",
        fontsize=11, fontweight="bold", pad=10)

    # Custom legend untuk kelas
    handles, labels = ax.get_legend_handles_labels()
    cls_patches = [mpatches.Patch(color=c, label=cls.capitalize())
                   for c, cls in zip(CLASS_COLORS[:n], classes)]
    ax.legend(handles=handles + cls_patches,
              labels=labels + [c.capitalize() for c in classes],
              fontsize=9, framealpha=0.85, ncol=2)
    ax.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "viz_v3_eval_per_class_f1.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_v3_eval_per_class_f1.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 3 — 3 VERSI PERBANDINGAN (UNGGULAN)
# =============================================================================
def viz_3versions_comparison(metrics_v3, f1s_v3, prcs_v3, recs_v3,
                              classes, output_dir):
    v3 = {
        "f1_macro"    : metrics_v3["f1_macro"],
        "f1_weighted" : metrics_v3["f1_weighted"],
        "accuracy"    : metrics_v3["accuracy"],
        "f1_neg"      : f1s_v3[0] if len(f1s_v3)>0 else 0,
        "f1_neu"      : f1s_v3[1] if len(f1s_v3)>1 else 0,
        "f1_pos"      : f1s_v3[2] if len(f1s_v3)>2 else 0,
        "label"       : "v3.0 Test\n(SMOTE Partial)",
    }

    versions   = [V1, V2, v3]
    ver_colors = [VIZ_V1, VIZ_V2, VIZ_V3]
    meta_keys  = ["f1_macro", "f1_weighted", "accuracy"]
    meta_names = ["F1-Macro", "F1-Weighted", "Accuracy"]
    cls_keys   = ["f1_neg", "f1_neu", "f1_pos"]
    cls_names  = ["F1 Negative", "F1 Neutral\n[TARGET]", "F1 Positive"]

    x = np.arange(len(meta_names))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.5))
    fig.patch.set_facecolor(VIZ_BG)

    # -- Overall metrics --
    ax1.set_facecolor(VIZ_BG)
    for vi, (ver, col) in enumerate(zip(versions, ver_colors)):
        off  = (vi - 1) * w
        vals = [ver[k] for k in meta_keys]
        bars = ax1.bar(x + off, vals, w,
                       label=ver["label"], color=col, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                     f"{val*100:.1f}%",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Anotasi gain v3 vs v1
    for i, k in enumerate(meta_keys):
        gain = v3[k] - V1[k]
        col  = "#27ae60" if gain >= 0 else "#e74c3c"
        ax1.text(x[i] + w, v3[k] + 0.045, f"{gain*100:+.1f}%",
                 ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", color=col)

    ax1.set_xticks(x)
    ax1.set_xticklabels(meta_names, fontsize=11)
    ax1.set_ylabel("Test Score (X_test Original)", fontsize=11)
    ax1.set_ylim(0, 1.20)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    ax1.set_title(
        "Perbandingan Test Score: v1.0 / v2.0 / v3.0\n"
        "Angka = gain/loss v3.0 vs v1.0 | Semua di X_test ORIGINAL yang sama",
        fontsize=10, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, framealpha=0.85)
    ax1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax1.spines[["top","right"]].set_visible(False)

    # -- F1 per kelas --
    ax2.set_facecolor(VIZ_BG)
    x2 = np.arange(len(cls_names))
    for vi, (ver, col) in enumerate(zip(versions, ver_colors)):
        off  = (vi - 1) * w
        vals = [ver[k] for k in cls_keys]
        bars = ax2.bar(x2 + off, vals, w,
                       label=ver["label"], color=col, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                     f"{val*100:.1f}%",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Anotasi gain Neutral (kelas target)
    neu_gain = v3["f1_neu"] - V1["f1_neu"]
    col_neu  = "#27ae60" if neu_gain >= 0 else "#e74c3c"
    ax2.text(x2[1] + w, v3["f1_neu"] + 0.055,
             f"Neutral\n{neu_gain*100:+.1f}%",
             ha="center", va="bottom", fontsize=9,
             fontweight="bold", color=col_neu)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cls_names, fontsize=10)
    ax2.set_ylabel("F1-Score per Kelas", fontsize=11)
    ax2.set_ylim(0, 1.25)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    ax2.set_title(
        "F1 per Kelas: v1.0 / v2.0 / v3.0\n"
        "Neutral = KELAS TARGET SMOTE | v2.0 gagal turun ke 4.3%\n"
        "v3.0 harusnya lebih baik dengan SMOTE partial",
        fontsize=10, fontweight="bold", pad=10)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax2.spines[["top","right"]].set_visible(False)

    # Tentukan verdict
    if v3["f1_neu"] > V1["f1_neu"] and v3["f1_macro"] >= V1["f1_macro"] - 0.02:
        verdict = f"[BERHASIL] SMOTE Partial meningkatkan Neutral F1 " \
                  f"{V1['f1_neu']*100:.1f}% -> {v3['f1_neu']*100:.1f}%!"
        verdict_color = "#27ae60"
    elif v3["f1_neu"] > V1["f1_neu"]:
        verdict = f"[SEBAGIAN] Neutral naik tapi F1-Macro turun. " \
                  f"Pertimbangkan threshold tuning."
        verdict_color = "#f39c12"
    elif v3["f1_macro"] > V1["f1_macro"]:
        verdict = f"[CAMPURAN] F1-Macro naik tapi Neutral tidak. " \
                  f"Coba IndoBERT embedding."
        verdict_color = "#f39c12"
    else:
        verdict = f"[PERLU PERBAIKAN] v3.0 belum ungguli v1.0. " \
                  f"Coba IndoBERT/IndoBERTweet."
        verdict_color = "#e74c3c"

    fig.suptitle(
        f"[PERBANDINGAN UTAMA] v1.0 vs v2.0 vs v3.0 — Test Set ORIGINAL\n"
        f"F1-Macro: v1={V1['f1_macro']*100:.2f}% | v2={V2['f1_macro']*100:.2f}% | "
        f"v3={v3['f1_macro']*100:.2f}%\n"
        f"Neutral: v1={V1['f1_neu']*100:.2f}% | v2={V2['f1_neu']*100:.2f}% | "
        f"v3={v3['f1_neu']*100:.2f}%",
        fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_v3_eval_3versions_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_v3_eval_3versions_comparison.png ({os.path.getsize(path)//1024} KB) <- UNGGULAN")
    return path, verdict, verdict_color


# =============================================================================
# VIZ 4 — ROC CURVE
# =============================================================================
def viz_roc_curve(y_test, y_proba, classes, output_dir):
    n = len(classes)
    y_bin = label_binarize(y_test, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    auc_scores = []
    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_proba[:,i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{cls.capitalize()} AUC={roc_auc:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.04, color=color)

    # Macro avg
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_bin[:,i], y_proba[:,i])[0] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:,i], y_proba[:,i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="#2c3e50", linewidth=2.5, linestyle="--",
            label=f"Macro-Avg AUC={macro_auc:.4f}")
    ax.plot([0,1],[0,1],"k:",linewidth=1.2,alpha=0.5,label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — v3.0 (SMOTE Partial)\nMacro AUC={macro_auc:.4f}",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.85)
    ax.grid(color=VIZ_GRID_CLR, linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "viz_v3_eval_roc_curve.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_v3_eval_roc_curve.png ({os.path.getsize(path)//1024} KB)")
    return path, macro_auc, auc_scores


# =============================================================================
# VIZ 5 — PR CURVE
# =============================================================================
def viz_pr_curve(y_test, y_proba, classes, output_dir):
    n = len(classes)
    y_bin = label_binarize(y_test, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(VIZ_BG)
    ax.set_facecolor(VIZ_BG)

    ap_scores = []
    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:,i], y_proba[:,i])
        ap = average_precision_score(y_bin[:,i], y_proba[:,i])
        ap_scores.append(ap)
        baseline_p = y_bin[:,i].mean()
        ax.plot(rec, prec, color=color, linewidth=2.2,
                label=f"{cls.capitalize()} AP={ap:.4f} (base={baseline_p:.3f})")
        ax.fill_between(rec, prec, alpha=0.04, color=color)
        ax.axhline(baseline_p, color=color, linewidth=0.8, linestyle=":", alpha=0.5)

    mean_ap = np.mean(ap_scores)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curve — v3.0\nMean AP={mean_ap:.4f}",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(color=VIZ_GRID_CLR, linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "viz_v3_eval_pr_curve.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_v3_eval_pr_curve.png ({os.path.getsize(path)//1024} KB)")
    return path, mean_ap, ap_scores


# =============================================================================
# VIZ 6 — CONFIDENCE DISTRIBUSI per KELAS
# =============================================================================
def viz_confidence_dist(y_test, y_pred, y_proba, classes, output_dir):
    n = len(classes)
    max_proba = y_proba.max(axis=1)
    correct   = (y_pred == y_test)

    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    fig.patch.set_facecolor(VIZ_BG)

    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        ax = axes[i]
        ax.set_facecolor(VIZ_BG)
        mask = (y_pred == i)
        conf_c = max_proba[mask & correct]
        conf_w = max_proba[mask & ~correct]
        bins   = np.linspace(0, 1, 25)
        if len(conf_c) > 0:
            ax.hist(conf_c, bins=bins, color=color, alpha=0.70,
                    label=f"Benar (n={len(conf_c)})", edgecolor="white")
        if len(conf_w) > 0:
            ax.hist(conf_w, bins=bins, color="#c0392b", alpha=0.55,
                    label=f"Salah (n={len(conf_w)})", edgecolor="white")
        tot = mask.sum()
        acc = len(conf_c)/tot if tot>0 else 0
        mn  = max_proba[mask].mean() if tot>0 else 0
        ax.axvline(0.5, color="#2c3e50", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(
            f"Pred: {cls.capitalize()}\n"
            f"n={tot} | Acc={acc*100:.1f}% | Mean={mn:.3f}",
            fontsize=10, fontweight="bold", color=color, pad=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(0, 1)

    fig.suptitle(
        f"Confidence Distribution — v3.0 (SMOTE Partial)\n"
        f"Mean max confidence: {max_proba.mean():.4f}",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "viz_v3_eval_confidence_dist.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_v3_eval_confidence_dist.png ({os.path.getsize(path)//1024} KB)")
    return path


# =============================================================================
# VIZ 7 — SUMMARY DASHBOARD (UNGGULAN)
# =============================================================================
def viz_summary_dashboard(metrics, cm, classes, f1s, macro_auc,
                           verdict, verdict_color, output_dir):
    n = len(classes)
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(VIZ_BG)
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.42)

    # -- Scorecard row -------------------------------------------------------
    score_data = [
        ("F1-Macro\n(Test v3.0)",   metrics["f1_macro"],
         V1["f1_macro"], VIZ_V3),
        ("F1-Weighted\n(Test v3.0)", metrics["f1_weighted"],
         V1["f1_weighted"], "#16a085"),
        ("Accuracy\n(Test v3.0)",   metrics["accuracy"],
         V1["accuracy"], "#8e44ad"),
        ("Neutral F1\n(Test v3.0)", f1s[1] if len(f1s)>1 else 0,
         V1["f1_neu"], VIZ_NEU),
        ("Macro AUC\n(Test v3.0)",  macro_auc, None, "#d35400"),
    ]
    for col_idx, (name, v3_val, v1_val, color) in enumerate(score_data):
        ax_s = fig.add_subplot(gs[0, col_idx])
        for sp in ax_s.spines.values():
            sp.set_visible(True)
            sp.set_color(color)
            sp.set_linewidth(2.5)
        ax_s.text(0.5, 0.72, f"{v3_val*100:.2f}%",
                  transform=ax_s.transAxes,
                  fontsize=20, fontweight="bold",
                  ha="center", va="center", color=color)
        ax_s.text(0.5, 0.44, name, transform=ax_s.transAxes,
                  fontsize=9, ha="center", va="center",
                  color="#212529", fontweight="bold")
        if v1_val is not None:
            gain = v3_val - v1_val
            gc   = "#27ae60" if gain >= 0 else "#e74c3c"
            ax_s.text(0.5, 0.16,
                      f"v1.0: {v1_val*100:.2f}%  {gain*100:+.2f}%",
                      transform=ax_s.transAxes,
                      fontsize=8, ha="center", va="center", color=gc)
        ax_s.axis("off")

    # -- Confusion Matrix ----------------------------------------------------
    ax_cm = fig.add_subplot(gs[1, 0:2])
    ax_cm.set_facecolor(VIZ_BG)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            txt_c = "white" if cm_norm[i,j] > 0.55 else "black"
            ax_cm.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                       ha="center", va="center", fontsize=10.5,
                       fontweight="bold", color=txt_c)
    ax_cm.set_xticks(range(n)); ax_cm.set_yticks(range(n))
    ax_cm.set_xticklabels([c.capitalize() for c in classes], fontsize=9.5)
    ax_cm.set_yticklabels([c.capitalize() for c in classes], fontsize=9.5)
    ax_cm.set_xlabel("Prediksi", fontsize=10)
    ax_cm.set_ylabel("Aktual", fontsize=10)
    ax_cm.set_title("Confusion Matrix v3.0\n(Test Set Original)",
                    fontsize=11, fontweight="bold")

    # -- F1 per kelas: v1 / v2 / v3 -----------------------------------------
    ax_f1 = fig.add_subplot(gs[1, 2:5])
    ax_f1.set_facecolor(VIZ_BG)
    x2   = np.arange(n)
    w    = 0.25
    v1_f = [V1["f1_neg"], V1["f1_neu"], V1["f1_pos"]]
    v2_f = [V2["f1_neg"], V2["f1_neu"], V2["f1_pos"]]
    v3_f = list(f1s[:n])

    for vi, (vals, col, lab) in enumerate([
        (v1_f, VIZ_V1, "v1.0 (no SMOTE)"),
        (v2_f, VIZ_V2, "v2.0 (SMOTE full x14)"),
        (v3_f, VIZ_V3, "v3.0 (SMOTE partial)"),
    ]):
        off  = (vi - 1) * w
        bars = ax_f1.bar(x2 + off, vals, w, label=lab,
                         color=col, alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax_f1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                       f"{val*100:.1f}%",
                       ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_f1.set_xticks(x2)
    ax_f1.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
    ax_f1.set_ylim(0, 1.20)
    ax_f1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v*100:.0f}%"))
    ax_f1.set_title("F1 per Kelas: v1.0 / v2.0 / v3.0 — Test Score",
                    fontsize=11, fontweight="bold")
    ax_f1.legend(fontsize=9, framealpha=0.85)
    ax_f1.grid(axis="y", color=VIZ_GRID_CLR, linewidth=0.7)
    ax_f1.spines[["top","right"]].set_visible(False)

    # -- Verdict & Rekomendasi -----------------------------------------------
    ax_rec = fig.add_subplot(gs[2, :])
    ax_rec.set_facecolor("white")
    for sp in ax_rec.spines.values():
        sp.set_visible(True); sp.set_color("#dee2e6")
    ax_rec.axis("off")

    neu_gain = f1s[1] - V1["f1_neu"] if len(f1s)>1 else 0
    macro_gain = metrics["f1_macro"] - V1["f1_macro"]

    lines = [
        (f"VERDICT: {verdict}", verdict_color, 12, True),
        ("", "#aaa", 4, False),
        (f"F1-Macro: v1={V1['f1_macro']*100:.2f}% | "
         f"v2={V2['f1_macro']*100:.2f}% | v3={metrics['f1_macro']*100:.2f}%  "
         f"(gain v3 vs v1: {macro_gain*100:+.2f}%)",
         "#34495e", 11, False),
        (f"Neutral F1: v1={V1['f1_neu']*100:.2f}% | "
         f"v2={V2['f1_neu']*100:.2f}% | v3={f1s[1]*100:.2f}%  "
         f"(gain v3 vs v1: {neu_gain*100:+.2f}%)",
         "#f39c12", 11, True),
        ("", "#aaa", 4, False),
        ("NEXT STEPS:", "#212529", 11, True),
    ]

    if neu_gain > 0.05 and macro_gain >= -0.02:
        lines += [
            ("  [1] v3.0 berhasil! Deploy model ini. Lakukan threshold tuning "
             "untuk Neutral jika precision/recall perlu diseimbangkan.",
             "#27ae60", 10, False),
            ("  [2] Untuk peningkatan lebih lanjut: ganti SBERT dengan IndoBERT "
             "(indobenchmark/indobert-base-p2).",
             "#2980b9", 10, False),
            ("  [3] Jika dataset bisa diperluas: tambah data Neutral yang benar-benar real.",
             "#8e44ad", 10, False),
        ]
    elif neu_gain <= 0:
        lines += [
            ("  [1] SMOTE partial masih kurang efektif untuk Neutral. "
             "Penyebab: embedding SBERT mungkin tidak cukup diskriminatif untuk Neutral.",
             "#e74c3c", 10, True),
            ("  [2] Ganti embedding: coba IndoBERT/IndoBERTweet. "
             "Literatur: F1-Macro ~75-82% untuk Indonesian sentiment.",
             "#2980b9", 10, False),
            ("  [3] Alternatif: Focal Loss SVM tidak tersedia langsung, "
             "coba XGBoost/LightGBM dengan scale_pos_weight.",
             "#8e44ad", 10, False),
            ("  [4] Audit kualitas label: apakah anotasi Neutral konsisten?",
             "#f39c12", 10, False),
        ]
    else:
        lines += [
            ("  [1] F1-Macro naik tapi Neutral belum membaik. "
             "Coba threshold tuning: prediksi Neutral jika P(neutral) > 0.15.",
             "#f39c12", 10, False),
            ("  [2] Ganti ke IndoBERT embedding untuk representasi lebih baik.",
             "#2980b9", 10, False),
        ]

    y_pos = 0.94
    for text, color, size, bold in lines:
        if not text:
            y_pos -= 0.025
            continue
        ax_rec.text(0.012, y_pos, text,
                    transform=ax_rec.transAxes,
                    fontsize=size, color=color,
                    fontweight="bold" if bold else "normal",
                    va="top")
        y_pos -= size * 0.010

    fig.suptitle(
        f"[DASHBOARD] EVALUASI v3.0 (SMOTE Partial) — SVM + SBERT\n"
        f"Test Set: {cm.sum()} sampel ORIGINAL | "
        f"v3.0: F1-Macro={metrics['f1_macro']*100:.2f}% | "
        f"Neutral={f1s[1]*100:.2f}% ({neu_gain*100:+.2f}% vs v1.0)",
        fontsize=13, fontweight="bold", y=1.01)
    path = os.path.join(output_dir, "viz_v3_eval_summary_dashboard.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=VIZ_BG)
    plt.close()
    log_ok(f"viz_v3_eval_summary_dashboard.png ({os.path.getsize(path)//1024} KB) <- UNGGULAN")
    return path


# =============================================================================
# MAIN
# =============================================================================
def main():
    global log_lines
    t_start   = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    banner = (
        "\n" + "=" * 65 + "\n"
        "  EVALUASI FINAL v3.0 (SMOTE PARTIAL) — SVM + SBERT\n"
        f"  Start  : {timestamp}\n"
        "  X_TEST : data ORIGINAL (imbalanced) — tidak di-SMOTE!\n"
        f"  Ref v1.0: F1-Macro={V1['f1_macro']*100:.2f}% | "
        f"Neutral={V1['f1_neu']*100:.2f}%\n"
        f"  Ref v2.0: F1-Macro={V2['f1_macro']*100:.2f}% | "
        f"Neutral={V2['f1_neu']*100:.2f}% (GAGAL)\n"
        "  TUJUAN : v3.0 harus lebih baik dari v2.0, dan ideally v1.0\n"
        + "=" * 65
    )
    print(C_HEAD + banner + C_RST)
    log_lines.append(banner)

    # =========================================================================
    # A: VERIFIKASI FILE
    # =========================================================================
    print_section("TAHAP A: VERIFIKASI FILE")
    for path, name in [
        (IN_BEST_MODEL, "svm_sbert_smote_v3_best_model.joblib"),
        (IN_X_TEST,     "X_test_emb.npy"),
        (IN_Y_TEST,     "y_test.npy"),
        (IN_CLASSES,    "label_classes.npy"),
    ]:
        if not os.path.exists(path):
            log_err(f"Tidak ditemukan: {path}")
            return
        log_ok(f"{name} ({os.path.getsize(path)//1024} KB)")

    # =========================================================================
    # B: LOAD DATA
    # =========================================================================
    print_section("TAHAP B: LOAD MODEL + TEST DATA")

    model   = joblib.load(IN_BEST_MODEL)
    X_test  = np.load(IN_X_TEST).astype(np.float32)
    y_test  = np.load(IN_Y_TEST)
    classes = np.load(IN_CLASSES, allow_pickle=True).tolist()
    n_test  = X_test.shape[0]
    n_cls   = len(classes)

    log_ok(f"Model: {type(model).__name__}")
    log_ok(f"X_test: {X_test.shape} | y_test: {y_test.shape}")
    log_ok(f"Classes: {classes}")

    log("\n  Distribusi X_test (ORIGINAL):")
    supports = []
    for i, cls in enumerate(classes):
        cnt = int((y_test == i).sum())
        supports.append(cnt)
        log(f"    {cls:<12} | {cnt:>4} ({cnt/n_test*100:.1f}%)")

    has_proba = hasattr(model, "predict_proba")

    # =========================================================================
    # C: PREDIKSI
    # =========================================================================
    print_section("TAHAP C: PREDIKSI")
    t0 = time.time()
    y_pred = model.predict(X_test)
    dur    = time.time() - t0
    log_ok(f"{n_test} sampel dalam {dur:.3f}s ({n_test/dur:.0f} sampel/s)")

    y_proba = None
    if has_proba:
        y_proba = model.predict_proba(X_test)
        log_ok(f"Proba: {y_proba.shape} | mean max = {y_proba.max(axis=1).mean():.4f}")

    # =========================================================================
    # D: METRIK
    # =========================================================================
    print_section("TAHAP D: METRIK EVALUASI")

    metrics = {
        "accuracy"          : accuracy_score(y_test, y_pred),
        "f1_macro"          : f1_score(y_test, y_pred, average="macro",    zero_division=0),
        "f1_weighted"       : f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_macro"   : precision_score(y_test, y_pred, average="macro",    zero_division=0),
        "recall_macro"      : recall_score(y_test,    y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted"   : recall_score(y_test,    y_pred, average="weighted", zero_division=0),
    }

    macro_auc  = 0.0
    auc_scores = [0.0]*n_cls
    ap_scores  = [0.0]*n_cls
    mean_ap    = 0.0

    if y_proba is not None:
        try:
            y_bin = label_binarize(y_test, classes=list(range(n_cls)))
            macro_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            metrics["roc_auc_macro"] = macro_auc
            for i in range(n_cls):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:,i], y_proba[:,i])
                auc_scores[i]   = auc(fpr_i, tpr_i)
                ap_scores[i]    = average_precision_score(y_bin[:,i], y_proba[:,i])
            mean_ap = np.mean(ap_scores)
        except Exception as e:
            log_warn(f"AUC gagal: {e}")

    cm   = confusion_matrix(y_test, y_pred)
    f1s  = f1_score(y_test, y_pred, average=None, zero_division=0)
    prcs = precision_score(y_test, y_pred, average=None, zero_division=0)
    recs = recall_score(y_test, y_pred, average=None, zero_division=0)

    cr = classification_report(
        y_test, y_pred,
        target_names=[c.capitalize() for c in classes],
        zero_division=0)

    # Print ringkasan
    log("\n" + "-"*65)
    log("  HASIL TEST SET FINAL v3.0")
    log("-"*65)
    comparisons = [
        ("F1-Macro",    V1["f1_macro"], V2["f1_macro"], metrics["f1_macro"]),
        ("F1-Weighted", V1["f1_weighted"],V2["f1_weighted"],metrics["f1_weighted"]),
        ("Accuracy",    V1["accuracy"],  V2["accuracy"],  metrics["accuracy"]),
        ("F1 Negative", V1["f1_neg"],    V2["f1_neg"],    f1s[0] if len(f1s)>0 else 0),
        ("F1 Neutral",  V1["f1_neu"],    V2["f1_neu"],    f1s[1] if len(f1s)>1 else 0),
        ("F1 Positive", V1["f1_pos"],    V2["f1_pos"],    f1s[2] if len(f1s)>2 else 0),
    ]
    log(f"  {'Metrik':<18} {'v1.0':>9} {'v2.0':>9} {'v3.0':>9} {'Gain v3-v1':>12}")
    log(f"  {'-'*60}")
    for nm, v1, v2, v3 in comparisons:
        gain = v3 - v1
        flag = " [GAIN]" if gain>0.005 else (" [~]" if abs(gain)<=0.005 else " [TURUN]")
        log(f"  {nm:<18} {v1*100:>8.2f}% {v2*100:>8.2f}% {v3*100:>8.2f}% "
            f"{gain*100:>+10.2f}%{flag}")
    log(f"\n  Classification Report:\n{cr}")

    # Simpan report
    cr_path = os.path.join(OUTPUT_DIR, "v3_eval_classification_report.txt")
    with open(cr_path, "w", encoding="utf-8") as f:
        f.write(f"CLASSIFICATION REPORT v3.0 — SMOTE Partial\n{timestamp}\n{'='*60}\n\n")
        f.write(cr)
        f.write(f"\n\n{'='*60}\nPERBANDINGAN v1.0 / v2.0 / v3.0\n{'='*60}\n")
        f.write(f"{'Metrik':<18} {'v1.0':>9} {'v2.0':>9} {'v3.0':>9} {'Gain v3-v1':>12}\n")
        f.write(f"{'-'*60}\n")
        for nm, v1, v2, v3 in comparisons:
            gain = v3 - v1
            f.write(f"{nm:<18} {v1*100:>8.2f}% {v2*100:>8.2f}% {v3*100:>8.2f}% "
                    f"{gain*100:>+10.2f}%\n")
    log_ok("v3_eval_classification_report.txt tersimpan")

    # Simpan CSV
    rows = []
    for nm, v1, v2, v3 in comparisons:
        rows.append({
            "metrik": nm, "v1_pct": f"{v1*100:.4f}%",
            "v2_pct": f"{v2*100:.4f}%", "v3_pct": f"{v3*100:.4f}%",
            "gain_v3_vs_v1": f"{(v3-v1)*100:+.4f}%"
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "v3_eval_metrics_summary.csv"), index=False)
    log_ok("v3_eval_metrics_summary.csv tersimpan")

    # =========================================================================
    # E: VISUALISASI
    # =========================================================================
    print_section("TAHAP E: VISUALISASI")

    viz_confusion_matrix(cm, classes, metrics, OUTPUT_DIR)
    viz_per_class_f1(f1s, prcs, recs, classes, supports, OUTPUT_DIR)
    _, verdict, verdict_color = viz_3versions_comparison(
        metrics, f1s, prcs, recs, classes, OUTPUT_DIR)

    if y_proba is not None:
        _, macro_auc, auc_scores = viz_roc_curve(y_test, y_proba, classes, OUTPUT_DIR)
        _, mean_ap, ap_scores    = viz_pr_curve(y_test, y_proba, classes, OUTPUT_DIR)
        viz_confidence_dist(y_test, y_pred, y_proba, classes, OUTPUT_DIR)
    else:
        log_warn("Proba tidak tersedia — skip ROC/PR/Confidence")

    viz_summary_dashboard(metrics, cm, classes, f1s, macro_auc,
                          verdict, verdict_color, OUTPUT_DIR)

    # =========================================================================
    # F: PENUTUP
    # =========================================================================
    elapsed = time.time() - t_start
    th, rem = divmod(int(elapsed), 3600)
    tm, ts  = divmod(rem, 60)

    neu_gain   = f1s[1] - V1["f1_neu"] if len(f1s)>1 else 0
    macro_gain = metrics["f1_macro"] - V1["f1_macro"]

    closing = (
        "\n" + "=" * 65 + "\n"
        "  EVALUASI v3.0 SELESAI!\n"
        f"\n  PERBANDINGAN TEST SCORE:\n"
        f"  {'Metrik':<18} {'v1.0':>9} {'v2.0':>9} {'v3.0':>9} {'Gain':>9}\n"
    )
    for nm, v1, v2, v3 in comparisons:
        closing += (f"  {nm:<18} {v1*100:>8.2f}% {v2*100:>8.2f}% "
                    f"{v3*100:>8.2f}% {(v3-v1)*100:>+8.2f}%\n")
    closing += (
        f"\n  VERDICT: {verdict}\n"
        f"\n  Waktu: {th}h {tm}m {ts:02d}s\n"
        + "=" * 65
    )
    print(C_HEAD + closing + C_RST)
    log_lines.append(closing)

    log_path = os.path.join(OUTPUT_DIR, "v3_evaluation_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log_ok(f"Log: {log_path}")


if __name__ == "__main__":
    main()
