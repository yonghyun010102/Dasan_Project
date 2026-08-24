"""Evaluation metrics for DCSC (paper Sec. 5.1.1).

Detection is scored as an utterance-level binary decision: the prediction is 1
when the system modified the utterance and 0 when it returned it unchanged;
the reference is 1 when the transcription differs from the ground truth.

Correction is scored with exact match and word error rate, each additionally
*balanced* over the naturally error-free and erroneous subsets, because the
corpus is error-sparse and unbalanced averages hide the interesting behaviour.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .edits import word_wer

__all__ = ["correction_metrics", "detection_metrics", "evaluate_predictions", "token_detection_metrics"]


def detection_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def correction_metrics(
    stt: Sequence[str], gold: Sequence[str], final: Sequence[str]
) -> Dict[str, float]:
    """EM / Bal-EM / N-WER / E-WER / Bal-WER.

    ``N`` (normal) is the subset where the transcription already equals the
    ground truth, ``E`` (erroneous) is its complement.
    """
    stt = [str(x) for x in stt]
    gold = [str(x) for x in gold]
    final = [str(x) for x in final]

    is_error = np.array([s.strip() != g.strip() for s, g in zip(stt, gold)])
    exact = np.array([f.strip() == g.strip() for f, g in zip(final, gold)], dtype=float)
    wer = np.array([word_wer(g, f) for g, f in zip(gold, final)], dtype=float)

    def _mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else 0.0

    normal, erroneous = ~is_error, is_error
    n_em, e_em = _mean(exact, normal), _mean(exact, erroneous)
    n_wer, e_wer = _mean(wer, normal), _mean(wer, erroneous)

    return {
        "exact_match": float(exact.mean()),
        "balanced_exact_match": (n_em + e_em) / 2.0,
        "normal_exact_match": n_em,
        "error_exact_match": e_em,
        "normal_wer": n_wer,
        "error_wer": e_wer,
        "balanced_wer": (n_wer + e_wer) / 2.0,
        "n_normal": int(normal.sum()),
        "n_error": int(erroneous.sum()),
    }


def evaluate_predictions(
    stt: Sequence[str], gold: Sequence[str], final: Sequence[str]
) -> Dict[str, float]:
    """Full metric block used for every reported configuration."""
    y_true = [int(str(s).strip() != str(g).strip()) for s, g in zip(stt, gold)]
    y_pred = [int(str(s).strip() != str(f).strip()) for s, f in zip(stt, final)]
    metrics = detection_metrics(y_true, y_pred)
    metrics.update(correction_metrics(stt, gold, final))
    metrics["n_samples"] = int(len(stt))
    return metrics


def token_detection_metrics(
    true_masks: List[List[int]], pred_masks: List[List[int]]
) -> Dict[str, float]:
    """Token-level detector quality plus the utterance-level roll-up used for
    checkpoint selection (validation F1)."""
    flat_true = [t for seq in true_masks for t in seq]
    flat_pred = [p for seq in pred_masks for p in seq]
    utt_true = [int(any(t == 1 for t in seq)) for seq in true_masks]
    utt_pred = [int(any(p == 1 for p in seq)) for seq in pred_masks]

    out = {f"token_{k}": v for k, v in detection_metrics(flat_true, flat_pred).items()}
    out.update({f"utt_{k}": v for k, v in detection_metrics(utt_true, utt_pred).items()})
    exact = np.array(
        [float(t == p) for t, p in zip(true_masks, pred_masks)], dtype=float
    )
    is_err = np.array([bool(any(t == 1 for t in seq)) for seq in true_masks])
    out["mask_exact_match"] = float(exact.mean()) if len(exact) else 0.0
    out["mask_normal_exact_match"] = float(exact[~is_err].mean()) if (~is_err).any() else 0.0
    out["mask_error_exact_match"] = float(exact[is_err].mean()) if is_err.any() else 0.0
    out["mask_balanced_exact_match"] = (
        out["mask_normal_exact_match"] + out["mask_error_exact_match"]
    ) / 2.0
    return out
