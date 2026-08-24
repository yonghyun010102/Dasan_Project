"""Span-level corrector with a Korean encoder-decoder backbone (paper Sec. 4.2).

The model maps a (optionally context-augmented) transcription to the span-level
correction string ``S``. ``[SEP]`` and ``->`` are added to the vocabulary so the
output format is represented by dedicated learnable tokens.

Checkpoint selection follows the paper: the epoch with the lowest validation
**Bal-WER**, measured by generating span strings, applying them and scoring the
reconstructed utterances.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .edits import ARROW_TOKEN, SEP_TOKEN, apply_spans
from .metrics import correction_metrics

INPUT_COL = "corrector_input"
TARGET_COL = "corrector_target"

__all__ = ["train_seq2seq_corrector", "load_seq2seq_corrector", "Seq2SeqCorrector"]


def _prepare_backbone(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    added = tokenizer.add_tokens([SEP_TOKEN, ARROW_TOKEN])
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    return tokenizer, model


def _to_dataset(df: pd.DataFrame, tokenizer, max_source_len: int, max_target_len: int) -> Dataset:
    ds = Dataset.from_pandas(
        df[[INPUT_COL, TARGET_COL]].reset_index(drop=True), preserve_index=False
    )

    def encode(batch):
        model_inputs = tokenizer(
            [str(x) for x in batch[INPUT_COL]],
            max_length=max_source_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=[str(x) for x in batch[TARGET_COL]],
            max_length=max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(encode, batched=True, remove_columns=[INPUT_COL, TARGET_COL])


def _build_compute_metrics(tokenizer, val_df: pd.DataFrame, stt_col: str, gold_col: str):
    stt = val_df[stt_col].astype(str).tolist()
    gold = val_df[gold_col].astype(str).tolist()

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded = [
            d.replace(tokenizer.pad_token or "", "")
            .replace(tokenizer.eos_token or "", "")
            .strip()
            for d in decoded
        ]
        n = min(len(decoded), len(stt))
        final = [apply_spans(stt[i], decoded[i]) for i in range(n)]
        metrics = correction_metrics(stt[:n], gold[:n], final)
        return {k: float(v) for k, v in metrics.items()}

    return compute_metrics


def train_seq2seq_corrector(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str | Path,
    base_model: str = "paust/pko-t5-base",
    stt_col: str = "STT인식문",
    gold_col: str = "정답문",
    seed: int = 42,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    batch_size: int = 24,
    eval_batch_size: int = 24,
    max_source_len: int = 512,
    max_target_len: int = 128,
    generation_max_len: int = 128,
    num_beams: int = 1,
    save_total_limit: int = 1,
    val_max_samples: int = 0,
    keep_intermediate: bool = False,
) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if val_max_samples and val_max_samples < len(val_df):
        val_df = val_df.sample(n=val_max_samples, random_state=seed).reset_index(drop=True)

    tokenizer, model = _prepare_backbone(base_model)
    train_ds = _to_dataset(train_df, tokenizer, max_source_len, max_target_len)
    val_ds = _to_dataset(val_df, tokenizer, max_source_len, max_target_len)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=lr,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="balanced_wer",
        greater_is_better=False,
        save_total_limit=save_total_limit,
        predict_with_generate=True,
        generation_max_length=generation_max_len,
        generation_num_beams=num_beams,
        logging_steps=100,
        seed=seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=_build_compute_metrics(tokenizer, val_df, stt_col, gold_col),
    )
    trainer.train()
    metrics = trainer.evaluate()

    best_dir = out_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    if not keep_intermediate:
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def load_seq2seq_corrector(model_dir: str | Path, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.config.use_cache = True
    model.to(device).eval()
    return model, tokenizer


class Seq2SeqCorrector:
    """Greedy span-string generation."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        max_source_len: int = 512,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        batch_size: int = 32,
    ):
        self.model, self.tokenizer = load_seq2seq_corrector(model_dir, device)
        self.device = device
        self.max_source_len = max_source_len
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(self, inputs: Sequence[str]) -> List[str]:
        outputs: List[str] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = [str(x) for x in inputs[start : start + self.batch_size]]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_len,
            ).to(self.device)
            generated = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams
            )
            outputs.extend(
                t.strip() for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs
