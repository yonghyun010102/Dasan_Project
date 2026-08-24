"""DasanCallDial loading, dialogue-level splitting and utterance-level deduplication.

Reproduces the dataset construction of the paper (Sec. 3.1.2 / 3.1.3, Tab. 2-3):

1. read the dialogue-level corpus from the Excel workbook,
2. split at the **dialogue** level with a 0.80 / 0.05 / 0.15 ratio, so no dialogue
   is shared between subsets,
3. deduplicate overlapping utterances **within each partition**, which removes the
   error-free boilerplate that would otherwise dominate training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Canonical column / sheet names of the released workbook.
PRED_SHEET = "인식문"
GOLD_SHEET = "정답문"
STT_COL = "STT인식문"
GOLD_COL = "정답문"
KEY_COL = "KEY"
SPEAKER_COL = "rec_mode"   # tx = counselor, rx = customer (the actual speaker)
MODE_COL = "in_out"        # call direction, not a speaker label
ERROR_COL = "오류 여부"

SPLITS = ("train", "val", "test")

__all__ = [
    "PRED_SHEET", "GOLD_SHEET", "STT_COL", "GOLD_COL", "KEY_COL",
    "SPEAKER_COL", "MODE_COL", "ERROR_COL", "SPLITS",
    "load_dialogue_corpus", "split_by_dialogue", "deduplicate",
    "build_splits", "split_statistics", "save_splits", "load_splits",
]


def load_dialogue_corpus(xlsx: str | Path) -> pd.DataFrame:
    """Load the dialogue-level corpus and attach the binary error flag."""
    pred = pd.read_excel(xlsx, sheet_name=PRED_SHEET)
    gold = pd.read_excel(xlsx, sheet_name=GOLD_SHEET)
    if len(pred) != len(gold):
        raise ValueError(f"sheet length mismatch: {len(pred)} vs {len(gold)}")

    df = pred.copy()
    df[GOLD_COL] = gold[GOLD_COL].values
    df[STT_COL] = df[STT_COL].astype(str)
    df[GOLD_COL] = df[GOLD_COL].astype(str)
    df[ERROR_COL] = (
        df[STT_COL].str.strip() != df[GOLD_COL].str.strip()
    ).astype(int)
    # preserve the original ordering inside each dialogue
    df["utt_index"] = df.groupby(KEY_COL).cumcount()
    return df.reset_index(drop=True)


def split_by_dialogue(
    df: pd.DataFrame,
    split_seed: int = 77,
    test_size: float = 0.20,
    val_share_of_holdout: float = 0.25,
) -> Dict[str, pd.DataFrame]:
    """Group split by dialogue key, yielding a 0.80 / 0.05 / 0.15 partition.

    ``test_size`` carves off the 20 % hold-out, of which ``val_share_of_holdout``
    becomes validation (0.25 * 0.20 = 0.05) and the remainder test (0.15).
    With 1,974 dialogues this gives 1,579 / 98 / 297 dialogues.
    """
    keys = df[KEY_COL].unique()
    train_keys, holdout_keys = train_test_split(
        keys, test_size=test_size, random_state=split_seed
    )
    val_keys, test_keys = train_test_split(
        holdout_keys, test_size=(1.0 - val_share_of_holdout), random_state=split_seed
    )
    return {
        "train": df[df[KEY_COL].isin(train_keys)].copy(),
        "val": df[df[KEY_COL].isin(val_keys)].copy(),
        "test": df[df[KEY_COL].isin(test_keys)].copy(),
    }


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated (transcription, ground-truth) pairs inside one partition."""
    return df.drop_duplicates(subset=[STT_COL, GOLD_COL]).reset_index(drop=True)


def build_splits(
    xlsx: str | Path,
    split_seed: int = 77,
    test_size: float = 0.20,
    val_share_of_holdout: float = 0.25,
    dedup: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Return ``(dialogue_level, utterance_level)`` splits.

    The dialogue-level frames keep every utterance (needed for dialogue context
    augmentation); the utterance-level frames are deduplicated and are what the
    detector / corrector are trained and evaluated on.
    """
    df = load_dialogue_corpus(xlsx)
    dialogue_level = split_by_dialogue(
        df, split_seed=split_seed, test_size=test_size,
        val_share_of_holdout=val_share_of_holdout,
    )
    utterance_level = {
        name: (deduplicate(part) if dedup else part.reset_index(drop=True))
        for name, part in dialogue_level.items()
    }
    return dialogue_level, utterance_level


def _speaker_rows(part: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    col = SPEAKER_COL if SPEAKER_COL in part.columns else MODE_COL
    for value, grp in part.groupby(col):
        out[str(value)] = {
            "n_samples": int(len(grp)),
            "error_rate": round(100.0 * float(grp[ERROR_COL].mean()), 2),
        }
    return out


def split_statistics(
    dialogue_level: Dict[str, pd.DataFrame],
    utterance_level: Dict[str, pd.DataFrame],
) -> Dict[str, dict]:
    """Reproduce the numbers of Tab. 2 (and the speaker breakdown of Tab. 3)."""
    stats: Dict[str, dict] = {"dialogue_level": {}, "utterance_level": {}, "speaker": {}}

    all_dlg = pd.concat(dialogue_level.values())
    all_utt = pd.concat(utterance_level.values())
    for name, part in list(dialogue_level.items()) + [("total", all_dlg)]:
        n_dlg = int(part[KEY_COL].nunique())
        stats["dialogue_level"][name] = {
            "n_dialogues": n_dlg,
            "n_utterances": int(len(part)),
            "avg_turns": round(len(part) / max(n_dlg, 1), 2),
        }
    for name, part in list(utterance_level.items()) + [("total", all_utt)]:
        stats["utterance_level"][name] = {
            "n_samples": int(len(part)),
            "avg_chars": round(float(part[STT_COL].str.len().mean()), 2),
            "error_rate": round(100.0 * float(part[ERROR_COL].mean()), 2),
        }
        stats["speaker"][name] = _speaker_rows(part)
    return stats


def save_splits(utterance_level: Dict[str, pd.DataFrame], out_dir: str | Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in utterance_level.items():
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    return paths


def load_splits(
    split_dir: str | Path, names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    split_dir = Path(split_dir)
    out = {}
    for name in names or list(SPLITS):
        path = split_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path} (run scripts/prepare_data.py)")
        part = pd.read_csv(path, encoding="utf-8-sig")
        part[STT_COL] = part[STT_COL].astype(str)
        part[GOLD_COL] = part[GOLD_COL].astype(str)
        out[name] = part
    return out
