"""Token-level ASR error detector (paper Sec. 4.1).

A KoELECTRA encoder is fine-tuned as a token classifier that predicts, for every
token of the transcription, whether it must be repaired. Because clean tokens
vastly outnumber erroneous ones, the cross-entropy loss is weighted by
``lambda_error`` (8 in the paper). At inference an utterance is forwarded to the
corrector when at least one token fires above ``threshold``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .data import GOLD_COL, STT_COL
from .edits import word_error_mask
from .metrics import token_detection_metrics

__all__ = ["WeightedTokenTrainer", "train_detector", "load_detector", "DetectorPredictor"]


class WeightedTokenTrainer(Trainer):
    """Trainer applying a class weight to the positive (erroneous) token label."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss_fct = CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def _encode(tokenizer, examples, max_length: int, label_all_tokens: bool = True):
    enc = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    aligned = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = enc.word_ids(batch_index=i)
        previous = None
        row = []
        for word_idx in word_ids:
            if word_idx is None:
                row.append(-100)
            elif word_idx != previous:
                row.append(labels[word_idx] if word_idx < len(labels) else 0)
            elif label_all_tokens:
                # every sub-token of a word carries that word's label
                row.append(labels[word_idx] if word_idx < len(labels) else 0)
            else:
                row.append(-100)
            previous = word_idx
        aligned.append(row)
    enc["labels"] = aligned
    return enc


def _to_dataset(df: pd.DataFrame, tokenizer, max_length: int,
                label_all_tokens: bool = True) -> Dataset:
    tokens = [str(t).split() for t in df[STT_COL]]
    labels = [word_error_mask(s, g) for s, g in zip(df[STT_COL], df[GOLD_COL])]
    keep = [i for i, tk in enumerate(tokens) if len(tk) > 0]
    ds = Dataset.from_dict(
        {"tokens": [tokens[i] for i in keep], "labels": [labels[i] for i in keep]}
    )
    return ds.map(
        lambda ex: _encode(tokenizer, ex, max_length, label_all_tokens),
        batched=True,
        remove_columns=["tokens", "labels"],
    )


def _build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=2)
        true_masks, pred_masks = [], []
        for pr, lb in zip(preds, labels):
            true_masks.append([int(l) for p, l in zip(pr, lb) if l != -100])
            pred_masks.append([int(p) for p, l in zip(pr, lb) if l != -100])
        metrics = token_detection_metrics(true_masks, pred_masks)
        # `f1` drives checkpoint selection (paper: best validation F1)
        metrics["f1"] = metrics["utt_f1"]
        return metrics

    return compute_metrics


def train_detector(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str | Path,
    base_model: str = "monologg/koelectra-base-v3-discriminator",
    seed: int = 42,
    max_length: int = 128,
    epochs: int = 15,
    lr: float = 2e-5,
    batch_size: int = 24,
    eval_batch_size: int = 24,
    lambda_error: float = 8.0,
    save_total_limit: int = 1,
    fp16: bool = False,
    keep_intermediate: bool = False,
    label_all_tokens: bool = True,
    metric_for_best: str = "f1",
) -> Dict[str, float]:
    """Fine-tune the detector and save the best checkpoint under ``out_dir/best``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=2)

    train_ds = _to_dataset(train_df, tokenizer, max_length, label_all_tokens)
    val_ds = _to_dataset(val_df, tokenizer, max_length, label_all_tokens)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        save_total_limit=save_total_limit,
        logging_steps=100,
        seed=seed,
        report_to=[],
        fp16=fp16,
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=_build_compute_metrics(),
        class_weights=torch.tensor([1.0, float(lambda_error)]),
    )
    trainer.train()
    metrics = trainer.evaluate()

    # validation F1 is nearly flat across epochs while the precision/recall
    # balance moves a lot, so the whole trajectory is recorded
    history = [
        {k: v for k, v in rec.items() if k.startswith("eval_") or k == "epoch"}
        for rec in trainer.state.log_history
        if any(k.startswith("eval_") for k in rec)
    ]
    with open(out_dir / "val_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    best_dir = out_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    if not keep_intermediate:
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def load_detector(model_dir: str | Path, device: str = "cuda"):
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model.to(device).eval()
    return model, tokenizer


class DetectorPredictor:
    """Batched token-level inference with an explicit firing threshold."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        max_length: int = 128,
        batch_size: int = 64,
        threshold: float = 0.5,
    ):
        self.model, self.tokenizer = load_detector(model_dir, device)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.threshold = threshold

    @torch.no_grad()
    def predict_masks(self, texts: Sequence[str]) -> List[List[int]]:
        masks: List[List[int]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(t).split() for t in texts[start : start + self.batch_size]]
            batch = [b if b else [""] for b in batch]
            enc = self.tokenizer(
                batch,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc_on_device = {k: v.to(self.device) for k, v in enc.items()}
            probs = torch.softmax(self.model(**enc_on_device).logits, dim=-1)[..., 1]
            probs = probs.detach().cpu().numpy()
            for i in range(len(batch)):
                word_ids = enc.word_ids(batch_index=i)
                per_word: Dict[int, float] = {}
                for pos, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue
                    # a word fires if ANY of its sub-tokens fires
                    prob = float(probs[i, pos])
                    if prob > per_word.get(word_idx, -1.0):
                        per_word[word_idx] = prob
                mask = [
                    int(per_word.get(w, 0.0) >= self.threshold)
                    for w in range(len(batch[i]))
                ]
                masks.append(mask)
        return masks

    def predict_flags(self, texts: Sequence[str]) -> np.ndarray:
        """1 when the utterance should be forwarded to the corrector."""
        masks = self.predict_masks(texts)
        return np.array([int(any(m)) for m in masks], dtype=int)
