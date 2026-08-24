"""Variant routing, corrector-training-set curation and cascade inference.

Four configurations are supported, matching the progressive ablation of the
paper (Tab. 6):

======  =======  ==============  ==============  ==================================
name    context  target          detector gate   description
======  =======  ==============  ==============  ==================================
``u``   no       full sentence   no              utterance -> utterance rewrite
``s``   no       span string     no              utterance -> error spans
``con_s`` yes    span string     no              dialogue context + utterance -> spans
``full`` yes     span string     yes             **DCSC** (the proposed pipeline)
======  =======  ==============  ==============  ==================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from .context import ContextBuffer, build_corrector_input, format_line
from .data import ERROR_COL, GOLD_COL, KEY_COL, MODE_COL, SPEAKER_COL, STT_COL
from .edits import NO_ERROR, apply_spans, make_span_target
from .metrics import evaluate_predictions

INPUT_COL = "corrector_input"
TARGET_COL = "corrector_target"

VARIANTS = ("u", "s", "con_s", "full")

__all__ = [
    "VariantSpec", "get_variant", "VARIANTS",
    "build_corrector_frame", "curate_detector_gated_frame",
    "run_pipeline", "evaluate_pipeline",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_context: bool
    span_target: bool
    use_detector: bool
    description: str


_SPECS = {
    "u": VariantSpec("u", False, False, False, "utterance -> utterance rewrite"),
    "s": VariantSpec("s", False, True, False, "utterance -> error spans"),
    "con_s": VariantSpec("con_s", True, True, False, "context + utterance -> error spans"),
    "full": VariantSpec("full", True, True, True, "DCSC: detector-gated contextual span correction"),
}


def get_variant(name: str) -> VariantSpec:
    key = str(name).strip().lower()
    aliases = {"dcsc": "full", "utterance": "u", "span": "s", "context_span": "con_s", "con+s": "con_s"}
    key = aliases.get(key, key)
    if key not in _SPECS:
        raise ValueError(f"unknown variant '{name}'; choose from {sorted(_SPECS)} (aliases: {sorted(aliases)})")
    return _SPECS[key]


def _speaker_column(df: pd.DataFrame) -> Optional[str]:
    for col in (SPEAKER_COL, MODE_COL):
        if col in df.columns:
            return col
    return None


def _gold_context_map(dialogue_part: pd.DataFrame, context_n: int) -> Dict[tuple, List[str]]:
    """(KEY, utt_index) -> ground-truth history window (training-time context)."""
    speaker_col = _speaker_column(dialogue_part)
    mapping: Dict[tuple, List[str]] = {}
    for key, group in dialogue_part.sort_values([KEY_COL, "utt_index"]).groupby(KEY_COL, sort=False):
        buffer = ContextBuffer(context_n)
        for _, row in group.iterrows():
            mapping[(key, int(row["utt_index"]))] = list(buffer.window())
            buffer.append(row[GOLD_COL], row[speaker_col] if speaker_col else None)
    return mapping


def build_corrector_frame(
    utterance_part: pd.DataFrame,
    dialogue_part: pd.DataFrame,
    variant: VariantSpec,
    context_n: int = 10,
) -> pd.DataFrame:
    """Attach ``corrector_input`` / ``corrector_target`` to a utterance-level frame."""
    df = utterance_part.copy().reset_index(drop=True)
    speaker_col = _speaker_column(df)

    if variant.use_context:
        ctx_map = _gold_context_map(dialogue_part, context_n)
        contexts = [
            ctx_map.get((row[KEY_COL], int(row["utt_index"])), [])
            for _, row in df.iterrows()
        ]
    else:
        contexts = [[] for _ in range(len(df))]

    current = [
        format_line(row[STT_COL], row[speaker_col] if speaker_col else None)
        for _, row in df.iterrows()
    ]
    df[INPUT_COL] = [
        build_corrector_input(cur, ctx, use_context=variant.use_context)
        for cur, ctx in zip(current, contexts)
    ]
    if variant.span_target:
        df[TARGET_COL] = [
            make_span_target(s, g) for s, g in zip(df[STT_COL], df[GOLD_COL])
        ]
    else:
        df[TARGET_COL] = df[GOLD_COL].astype(str)
    return df


def curate_detector_gated_frame(
    frame: pd.DataFrame, detector_flags: Sequence[int], seed: int = 42,
    strategy: str = "fp_aware",
) -> pd.DataFrame:
    """Detector-gated corrector training set (paper Sec. 4.3).

    Collects (1) every genuinely erroneous utterance, (2) the utterances the
    detector wrongly flags as erroneous, and (3) a random error-free remainder,
    so the error-free half matches the erroneous half in size.
    """
    df = frame.copy().reset_index(drop=True)
    df["_det"] = np.asarray(detector_flags, dtype=int)

    erroneous = df[df[ERROR_COL] == 1]
    clean = df[df[ERROR_COL] == 0]
    if len(erroneous) == 0 or len(clean) == 0:
        return df.drop(columns=["_det"])

    quota = min(len(erroneous), len(clean))
    if len(erroneous) > quota:
        erroneous = erroneous.sample(n=quota, random_state=seed)

    if strategy == "balanced":
        # random error-free half, ignoring the detector's decisions
        selected_clean = clean.sample(n=quota, random_state=seed)
        curated = (
            pd.concat([erroneous, selected_clean], axis=0)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
            .drop(columns=["_det"])
        )
        return curated

    false_positives = clean[clean["_det"] == 1]
    take_fp = min(len(false_positives), quota)
    selected_clean = (
        false_positives.sample(n=take_fp, random_state=seed)
        if take_fp > 0
        else false_positives.iloc[0:0]
    )
    shortfall = quota - len(selected_clean)
    if shortfall > 0:
        remainder = clean.drop(index=selected_clean.index, errors="ignore")
        extra = remainder.sample(n=min(shortfall, len(remainder)), random_state=seed)
        selected_clean = pd.concat([selected_clean, extra], axis=0)

    if len(selected_clean) < len(erroneous):
        erroneous = erroneous.sample(n=len(selected_clean), random_state=seed)

    curated = (
        pd.concat([erroneous, selected_clean], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
        .drop(columns=["_det"])
    )
    return curated


class CorrectorLike(Protocol):
    def generate(self, inputs: Sequence[str]) -> List[str]:
        ...


def run_pipeline(
    dialogue_part: pd.DataFrame,
    variant: VariantSpec,
    corrector: CorrectorLike,
    detector_flags: Optional[Dict[tuple, int]] = None,
    context_n: int = 10,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Walk every dialogue in order and produce the corrected transcript.

    For the context-augmented variants the history is filled **autoregressively
    with previously corrected utterances**, because the ground-truth history is
    unavailable at inference time (paper Sec. 4.3 / 6.3); those variants are
    therefore decoded utterance by utterance. Context-free variants do not
    depend on the history and are decoded in batches.
    """
    speaker_col = _speaker_column(dialogue_part)
    ordered = dialogue_part.sort_values([KEY_COL, "utt_index"]).reset_index(drop=True)

    def gate_of(key, utt_index) -> int:
        if not variant.use_detector or detector_flags is None:
            return 1
        return int(detector_flags.get((key, int(utt_index)), 1))

    def finalize(stt: str, prediction: str) -> str:
        if variant.span_target:
            return apply_spans(stt, prediction)
        return prediction.strip() or stt

    records: List[dict] = []

    if not variant.use_context:
        # ---- context-free: one batched pass over all gate-open utterances ----
        for _, row in ordered.iterrows():
            stt = str(row[STT_COL])
            speaker = row[speaker_col] if speaker_col else None
            gate = gate_of(row[KEY_COL], row["utt_index"])
            records.append({
                KEY_COL: row[KEY_COL],
                "utt_index": int(row["utt_index"]),
                STT_COL: stt,
                GOLD_COL: str(row[GOLD_COL]),
                "speaker": speaker,
                "detector_flag": gate,
                "input": build_corrector_input(format_line(stt, speaker), (), use_context=False),
            })
        todo = [i for i, r in enumerate(records) if r["detector_flag"] == 1]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            outputs = corrector.generate([records[i]["input"] for i in chunk])
            for i, prediction in zip(chunk, outputs):
                records[i]["prediction"] = prediction
        for record in records:
            if record["detector_flag"] == 0:
                record["prediction"] = NO_ERROR if variant.span_target else record[STT_COL]
                record["final"] = record[STT_COL]
            else:
                record["final"] = finalize(record[STT_COL], record.get("prediction", ""))
    else:
        # ---- context-augmented: sequential decoding per dialogue ----
        for key, group in ordered.groupby(KEY_COL, sort=False):
            buffer = ContextBuffer(context_n)
            for _, row in group.iterrows():
                stt = str(row[STT_COL])
                speaker = row[speaker_col] if speaker_col else None
                gate = gate_of(key, row["utt_index"])
                record = {
                    KEY_COL: key,
                    "utt_index": int(row["utt_index"]),
                    STT_COL: stt,
                    GOLD_COL: str(row[GOLD_COL]),
                    "speaker": speaker,
                    "detector_flag": gate,
                }
                if gate == 0:
                    record["prediction"] = NO_ERROR if variant.span_target else stt
                    record["final"] = stt
                else:
                    corrector_input = build_corrector_input(
                        format_line(stt, speaker), buffer.window(), use_context=True
                    )
                    prediction = corrector.generate([corrector_input])[0]
                    record["prediction"] = prediction
                    record["final"] = finalize(stt, prediction)
                buffer.append(record["final"], speaker)
                records.append(record)

    out = pd.DataFrame(records)
    return out.drop(columns=[c for c in ["input"] if c in out.columns])


def evaluate_pipeline(
    predictions: pd.DataFrame,
    score_subset: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Score the pipeline output, optionally restricted to a deduplicated subset."""
    df = predictions
    if score_subset is not None:
        keys = set(
            zip(score_subset[KEY_COL].tolist(), score_subset["utt_index"].astype(int).tolist())
        )
        mask = [
            (k, i) in keys
            for k, i in zip(df[KEY_COL].tolist(), df["utt_index"].astype(int).tolist())
        ]
        df = df[pd.Series(mask, index=df.index)]
    return evaluate_predictions(
        df[STT_COL].tolist(), df[GOLD_COL].tolist(), df["final"].tolist()
    )
