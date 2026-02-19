# evaluator.py
# Metrics and k-fold utilities for nucleosome positioning evaluation.

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold


# ══════════════════════════════════════════════════════════════════════════════
# K-fold builder
# ══════════════════════════════════════════════════════════════════════════════

def build_kfold(data, labels, k=5, shuffle=True, seed=42):
    """
    Build stratified k-fold splits preserving the nucleosome/linker class ratio.

    Args:
        data:    np.ndarray  (N, 145, 12)
        labels:  np.ndarray  (N, 1)  — 1=nucleosomal, 0=linker
        k:       number of folds
        shuffle: whether to shuffle before splitting (passed through correctly)
        seed:    random seed for reproducibility

    Returns:
        list of dicts with keys X1_train, X1_test, y_train, y_test
    """
    # Flatten labels to 1-D for StratifiedKFold
    labels_1d = labels.ravel()

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed if shuffle else None)

    fold_list = []
    for train_idx, test_idx in skf.split(data, labels_1d):
        fold_list.append({
            "X1_train": data[train_idx],
            "X1_test":  data[test_idx],
            "y_train":  labels[train_idx],   # keep (N,1) shape for model
            "y_test":   labels[test_idx],
        })

    return fold_list


# ══════════════════════════════════════════════════════════════════════════════
# Prediction helper
# ══════════════════════════════════════════════════════════════════════════════

def pred2label(y_pred):
    """Convert sigmoid probabilities to hard 0/1 labels."""
    return np.round(np.clip(y_pred, 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# Keras-compatible metric functions
# (used as compile metrics so they must accept tensors)
# ══════════════════════════════════════════════════════════════════════════════

def precision(y_true, y_pred):
    """Precision = TP / (TP + FP)"""
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos      = K.round(K.clip(y_true, 0, 1))
    y_neg      = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    return tp / (tp + fp + K.epsilon())


def recall(y_true, y_pred):
    """Recall / Sensitivity = TP / (TP + FN)"""
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos      = K.round(K.clip(y_true, 0, 1))
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return tp / (tp + fn + K.epsilon())


def f1score(y_true, y_pred):
    """F1 = 2 * precision * recall / (precision + recall)"""
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos      = K.round(K.clip(y_true, 0, 1))
    y_neg      = 1 - y_pos
    tp  = K.sum(y_pos * y_pred_pos)
    fp  = K.sum(y_neg * y_pred_pos)
    fn  = K.sum(y_pos * y_pred_neg)
    prec = tp / (tp + fp + K.epsilon())
    rec  = tp / (tp + fn + K.epsilon())
    return 2 * prec * rec / (prec + rec + K.epsilon())


def acc(y_true, y_pred):
    """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos      = K.round(K.clip(y_true, 0, 1))
    y_neg      = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return (tp + tn) / (tp + tn + fp + fn + K.epsilon())


# ══════════════════════════════════════════════════════════════════════════════
# AUC (trapezoidal approximation, Keras-compatible)
# ══════════════════════════════════════════════════════════════════════════════

def _binary_PFA(y_true, y_pred, threshold):
    """Probability of False Alert at a given threshold."""
    y_bin = K.cast(y_pred >= threshold, "float32")
    N  = K.sum(1 - y_true)
    FP = K.sum(y_bin - y_bin * y_true)
    return FP / (N + K.epsilon())


def _binary_PTA(y_true, y_pred, threshold):
    """Probability of True Alert at a given threshold."""
    y_bin = K.cast(y_pred >= threshold, "float32")
    P  = K.sum(y_true)
    TP = K.sum(y_bin * y_true)
    return TP / (P + K.epsilon())


def aucScore(y_true, y_pred):
    """Approximate AUC via trapezoidal rule over 1000 thresholds."""
    thresholds = np.linspace(0, 1, 1000)
    ptas = tf.stack([_binary_PTA(y_true, y_pred, k) for k in thresholds], axis=0)
    pfas = tf.stack([_binary_PFA(y_true, y_pred, k) for k in thresholds], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    bin_sizes = -(pfas[1:] - pfas[:-1])
    return K.sum(ptas * bin_sizes, axis=0)