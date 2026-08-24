"""Span-level corrector with a decoder-only LLM backbone (paper Sec. 4.2, Tab. 11).

The transcription (optionally context-augmented) is wrapped in an
instruction-tuning prompt and the model is fine-tuned with LoRA to emit the
span-level correction string. Prompt tokens are masked out of the loss.

Checkpoint selection mirrors the paper: after every epoch the adapter is scored
on validation **Bal-WER** by generating span strings, and only the best adapter
is kept.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .edits import ARROW_TOKEN, NO_ERROR, SEP_TOKEN, apply_spans
from .metrics import correction_metrics

INPUT_COL = "corrector_input"
TARGET_COL = "corrector_target"

SYSTEM_PROMPT = f"""당신은 한국어 STT 후처리 교정기입니다.

입력: 교정할 문장이 주어집니다(앞선 대화 문맥이 함께 제공될 수 있습니다).
출력: 아래 형식의 span 교정 문자열만 출력하세요.

규칙:
1) 문장이 완전히 올바르면 정확히 "{NO_ERROR}"만 출력하세요.
2) 오류가 있으면 (오류 span) {ARROW_TOKEN} (정정 span) 형태로 작성하세요.
3) 여러 개면 " {SEP_TOKEN} " 로 구분해 이어서 출력하세요.
4) 삭제만 필요한 경우는 출력하지 마세요.
5) 삽입은 바로 앞 단어를 anchor로 잡아 "anchor {ARROW_TOKEN} anchor 삽입span" 으로 표현하세요.
6) 설명/따옴표/불릿/JSON 없이 span 교정 문자열만 출력하세요.
"""

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

__all__ = ["train_llm_corrector", "LLMCorrector", "build_prompt"]


def _prepare_tokenizer(tokenizer, padding_side: str = "right"):
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(tokenizer, corrector_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(corrector_input).strip()},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT.strip()}\n\n"
        f"[USER]\n{str(corrector_input).strip()}\n\n[ASSISTANT]\n"
    )


class SpanSFTDataset(TorchDataset):
    """Prompt + target with the prompt region masked out of the labels."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = [build_prompt(tokenizer, x) for x in df[INPUT_COL]]
        self.targets = [str(x).strip() for x in df[TARGET_COL]]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        prompt_ids = self.tokenizer(
            self.prompts[idx], add_special_tokens=False
        )["input_ids"]
        target_ids = self.tokenizer(
            self.targets[idx], add_special_tokens=False
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            target_ids = target_ids + [self.tokenizer.eos_token_id]

        # keep the tail of the prompt if the pair is too long
        budget = self.max_length - len(target_ids)
        if budget < 1:
            target_ids = target_ids[: self.max_length - 1]
            budget = self.max_length - len(target_ids)
        prompt_ids = prompt_ids[-budget:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        return {"input_ids": input_ids, "labels": labels}


class CausalCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


@torch.no_grad()
def _generate_spans(
    model,
    tokenizer,
    inputs: Sequence[str],
    device,
    max_prompt_len: int = 768,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> List[str]:
    tokenizer = _prepare_tokenizer(tokenizer, padding_side="left")
    was_training = model.training
    model.eval()
    outputs: List[str] = []
    for start in range(0, len(inputs), batch_size):
        prompts = [build_prompt(tokenizer, x) for x in inputs[start : start + batch_size]]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=False,
        ).to(device)
        generated = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        for i in range(len(prompts)):
            new_tokens = generated[i, enc["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(text.replace("\n", " ").strip() or NO_ERROR)
    if was_training:
        model.train()
    _prepare_tokenizer(tokenizer, padding_side="right")
    return outputs


class BestBalWerCallback(TrainerCallback):
    """Score the adapter on validation Bal-WER each epoch and keep the best."""

    def __init__(self, trainer_ref, val_df, tokenizer, best_dir: Path,
                 stt_col: str, gold_col: str, max_prompt_len: int, gen_batch_size: int):
        self.trainer_ref = trainer_ref
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.best_dir = Path(best_dir)
        self.stt_col, self.gold_col = stt_col, gold_col
        self.max_prompt_len = max_prompt_len
        self.gen_batch_size = gen_batch_size
        self.best_score: Optional[float] = None
        self.history: List[Dict[str, float]] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref()
        if trainer is None or not trainer.is_world_process_zero():
            return control
        model = trainer.model
        spans = _generate_spans(
            model, self.tokenizer, self.val_df[INPUT_COL].tolist(),
            model.device, self.max_prompt_len, 128, self.gen_batch_size,
        )
        stt = self.val_df[self.stt_col].astype(str).tolist()
        gold = self.val_df[self.gold_col].astype(str).tolist()
        final = [apply_spans(s, p) for s, p in zip(stt, spans)]
        metrics = correction_metrics(stt, gold, final)
        score = metrics["balanced_wer"]
        self.history.append({"epoch": float(state.epoch or 0), **{k: float(v) for k, v in metrics.items()}})
        print(f"[llm-corrector] epoch {state.epoch:.2f} val Bal-WER={score:.4f}")
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(self.best_dir))
            self.tokenizer.save_pretrained(str(self.best_dir))
            print(f"[llm-corrector] new best adapter saved -> {self.best_dir}")
        return control


def train_llm_corrector(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str | Path,
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    stt_col: str = "STT인식문",
    gold_col: str = "정답문",
    seed: int = 42,
    epochs: int = 6,
    lr: float = 1e-4,
    per_device_batch_size: int = 3,
    grad_accum: int = 8,
    max_length: int = 768,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    val_max_samples: int = 300,
    gen_batch_size: int = 8,
    save_total_limit: int = 1,
) -> Dict[str, float]:
    import weakref

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if val_max_samples and val_max_samples < len(val_df):
        val_df = val_df.sample(n=val_max_samples, random_state=seed).reset_index(drop=True)
    else:
        val_df = val_df.reset_index(drop=True)

    tokenizer = _prepare_tokenizer(
        AutoTokenizer.from_pretrained(base_model, use_fast=True), "right"
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        # required so that gradients flow through checkpointed blocks into the
        # LoRA adapters (the frozen embedding output would otherwise be detached)
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        eval_strategy="no",
        save_strategy="no",
        save_total_limit=save_total_limit,
        logging_steps=20,
        warmup_ratio=0.03,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SpanSFTDataset(train_df, tokenizer, max_length),
        data_collator=CausalCollator(tokenizer),
    )
    callback = BestBalWerCallback(
        weakref.ref(trainer), val_df, tokenizer, out_dir / "best",
        stt_col, gold_col, max_length, gen_batch_size,
    )
    trainer.add_callback(callback)
    trainer.train()

    if trainer.is_world_process_zero():
        if callback.best_score is None:  # no epoch-end eval happened
            (out_dir / "best").mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir / "best"))
            tokenizer.save_pretrained(str(out_dir / "best"))
        with open(out_dir / "val_history.json", "w", encoding="utf-8") as f:
            json.dump(callback.history, f, ensure_ascii=False, indent=2)
    return {"best_balanced_wer": float(callback.best_score) if callback.best_score is not None else float("nan")}


class LLMCorrector:
    """Greedy span-string generation from a trained LoRA adapter."""

    def __init__(
        self,
        adapter_dir: str | Path,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        max_prompt_len: int = 768,
        max_new_tokens: int = 128,
        batch_size: int = 8,
        bf16: bool = True,
    ):
        self.tokenizer = _prepare_tokenizer(
            AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True), "left"
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16 if bf16 else torch.float32
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.config.use_cache = True
        self.model.to(device).eval()
        self.device = device
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def generate(self, inputs: Sequence[str]) -> List[str]:
        return _generate_spans(
            self.model, self.tokenizer, inputs, self.device,
            self.max_prompt_len, self.max_new_tokens, self.batch_size,
        )
