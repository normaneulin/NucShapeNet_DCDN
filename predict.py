# predict.py
# Evaluation script for Dual-Stream CNN nucleosome positioning model

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import math
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    accuracy_score, precision_score,
    f1_score, confusion_matrix,
)
from tensorflow.keras.models import load_model

import evaluator
from model_npcdn import DualStreamCNN

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Dual-Stream CNN Evaluation")
parser.add_argument("-pn", "--plot",  dest="plotName", type=str, default="DualStream-CNN")
parser.add_argument("-p",  "--path",  dest="path",     type=str, default="result")
parser.add_argument("-e",  "--experiments", dest="exp", default="Experiment_melanogaster")
parser.add_argument("-f",  "--foldName",    dest="foldName", default="folds.pickle")
args = parser.parse_args()

inPath   = args.path
expName  = args.exp
foldName = args.foldName
plotName = args.plotName

model_name = "npcdn"
modelPath  = os.path.join(inPath, expName, "models", model_name)
foldPath   = os.path.join(inPath, expName, foldName)

# ── Load folds ────────────────────────────────────────────────────────────────
if not os.path.exists(foldPath):
    print(f"[ERROR] Folds file not found: {foldPath}")
    sys.exit(1)

with open(foldPath, "rb") as fp:
    folds = pickle.load(fp)

print(f"[INFO] Loaded {len(folds)} folds from {foldPath}")

# ── Per-fold evaluation ───────────────────────────────────────────────────────
custom_objects = {
    "DualStreamCNN": DualStreamCNN,
    "precision":     evaluator.precision,
    "recall":        evaluator.recall,
    "f1score":       evaluator.f1score,
    "aucScore":      evaluator.aucScore,
    "acc":           evaluator.acc,
}

results = {k: [] for k in
           ["Accuracy", "Precision", "Sensitivity", "Specificity",
            "MCC", "AUC", "F1_Score"]}

all_tpr = []
all_fpr = []

for fold_idx, fold in enumerate(folds, start=1):
    model_file = os.path.join(modelPath, f"npcdn_best-fold{fold_idx}.keras")

    if not os.path.exists(model_file):
        print(f"[ERROR] Model not found for fold {fold_idx}: {model_file}")
        continue

    model = load_model(model_file, custom_objects=custom_objects)

    X_test = fold["X1_test"].astype("float32")
    y_test = fold["y_test"].astype("float32").ravel()

    # Raw probabilities and hard labels
    y_prob  = model.predict(X_test, verbose=0).ravel()
    y_pred  = np.round(np.clip(y_prob, 0, 1))

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc_  = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    f1_   = f1_score(y_test, y_pred, zero_division=0)
    auc_  = roc_auc_score(y_test, y_prob)

    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
    all_fpr.append(fpr_arr)
    all_tpr.append(tpr_arr)

    cm   = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    sens = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    spec = float(TN) / float(TN + FP) if (TN + FP) > 0 else 0.0

    denom = math.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    mcc   = (float(TP*TN) - float(FP*FN)) / denom if denom > 0 else 0.0

    results["Accuracy"].append(acc_)
    results["Precision"].append(prec)
    results["Sensitivity"].append(sens)
    results["Specificity"].append(spec)
    results["MCC"].append(mcc)
    results["AUC"].append(auc_)
    results["F1_Score"].append(f1_)

    print(f"  Fold {fold_idx}: acc={acc_:.4f}  sens={sens:.4f}  spec={spec:.4f}  "
          f"mcc={mcc:.4f}  auc={auc_:.4f}  f1={f1_:.4f}")

# ── Aggregate stats ───────────────────────────────────────────────────────────
metric_names = ["Accuracy", "Sensitivity", "Specificity", "MCC", "AUC", "F1_Score"]
means = [np.mean(results[m]) for m in metric_names]
stds  = [np.std(results[m])  for m in metric_names]

print("\n[SUMMARY]")
for name, mu, sigma in zip(metric_names, means, stds):
    print(f"  {name:15s}: {mu:.4f} ± {sigma:.4f}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
plot_dir = os.path.join(modelPath, "plot")
os.makedirs(plot_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(metric_names))
ax.bar(x, means, width=0.6)
ax.set_xticks(list(x))
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.15)
ax.set_title(f"{plotName} — 5-Fold Cross-Validation")
ax.set_ylabel("Score")

for i, (mu, sigma) in enumerate(zip(means, stds)):
    ax.text(i - 0.30, mu + 0.03, f"{mu:.4f}\n±{sigma:.4f}",
            color="blue", fontsize=8, fontweight="bold")

for fmt in ("png", "svg", "eps"):
    fig.savefig(os.path.join(plot_dir, f"{plotName}_eval.{fmt}"),
                format=fmt, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Bar chart saved to {plot_dir}")

# ── CSV ───────────────────────────────────────────────────────────────────────
csv_path = os.path.join(modelPath, "evaluation_per_fold.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold"] + metric_names)
    for idx in range(len(results["Accuracy"])):
        writer.writerow(
            [idx + 1] + [results[m][idx] for m in metric_names]
        )
    writer.writerow(
        ["Mean"] + [np.mean(results[m]) for m in metric_names]
    )
    writer.writerow(
        ["Std"]  + [np.std(results[m])  for m in metric_names]
    )

print(f"[INFO] Per-fold metrics saved to {csv_path}")