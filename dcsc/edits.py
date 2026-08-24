"""Edit-level primitives for DCSC.

Implements the three granularity refinements described in the paper:

* **token(word)-level error mask** -> detection targets (Sec. 4.1.1)
* **span-level correction string** -> corrector targets (Sec. 4.2.1)
* **span application** -> reconstruct the corrected utterance from a span string (Sec. 4.2.3)

The span string format is exactly::

    span_U -> span_G [SEP] span_U2 -> span_G2 ...

and the Korean phrase ``이상없음`` ("No Error") is emitted when an utterance
requires no modification.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np

SEP_TOKEN = "[SEP]"
ARROW_TOKEN = "->"
NO_ERROR = "이상없음"  # Korean for "No Error"

__all__ = [
    "SEP_TOKEN",
    "ARROW_TOKEN",
    "NO_ERROR",
    "word_error_mask",
    "extract_spans",
    "make_span_target",
    "apply_spans",
    "word_wer",
    "count_spans",
]


def word_error_mask(stt: str, gold: str) -> List[int]:
    """Binary error mask over the whitespace tokens of ``stt`` (Sec. 4.1.1).

    A token is masked with 1 when it is replaced or deleted with respect to
    ``gold``; for an insertion the token immediately preceding the insertion
    point is masked. All other tokens are 0.
    """
    stt_tokens = str(stt).split()
    gold_tokens = str(gold).split()
    labels = [0] * len(stt_tokens)

    opcodes = SequenceMatcher(None, stt_tokens, gold_tokens).get_opcodes()
    i = 0
    while i < len(opcodes):
        tag, i1, i2, _j1, _j2 = opcodes[i]

        if tag == "equal":
            # insertion right after an equal block -> mask the anchor token
            if i + 1 < len(opcodes) and opcodes[i + 1][0] == "insert" and (i2 - i1) > 0:
                anchor = i2 - 1
                if 0 <= anchor < len(labels):
                    labels[anchor] = 1
                i += 2
                continue
            i += 1
            continue

        if tag in ("replace", "delete"):
            for k in range(i1, i2):
                if 0 <= k < len(labels):
                    labels[k] = 1
            if i + 1 < len(opcodes) and opcodes[i + 1][0] == "insert" and (i2 - i1) > 0:
                anchor = i2 - 1
                if 0 <= anchor < len(labels):
                    labels[anchor] = 1
                i += 2
                continue
            i += 1
            continue

        # a leading insertion has no anchor to mask
        i += 1

    return labels


def extract_spans(stt: str, gold: str) -> List[Tuple[str, str]]:
    """Word-level (erroneous span, corrected span) pairs (Sec. 4.2.1).

    Consecutive erroneous tokens are merged into a single span. Deletions are
    not emitted (the corrector is replace-only) and insertions are expressed
    by anchoring on the preceding word.
    """
    stt_tokens = str(stt).split()
    gold_tokens = str(gold).split()

    opcodes = SequenceMatcher(None, stt_tokens, gold_tokens).get_opcodes()
    spans: List[Tuple[str, str]] = []

    i = 0
    while i < len(opcodes):
        tag, i1, i2, j1, j2 = opcodes[i]

        if tag == "equal":
            if i + 1 < len(opcodes) and opcodes[i + 1][0] == "insert" and (i2 - i1) > 0:
                _, _, _, nj1, nj2 = opcodes[i + 1]
                anchor = stt_tokens[i2 - 1]
                inserted = " ".join(gold_tokens[nj1:nj2]).strip()
                if inserted:
                    spans.append((anchor, f"{anchor} {inserted}".strip()))
                    i += 2
                    continue
            i += 1
            continue

        if tag == "replace":
            src = " ".join(stt_tokens[i1:i2]).strip()
            tgt = " ".join(gold_tokens[j1:j2]).strip()
            if i + 1 < len(opcodes) and opcodes[i + 1][0] == "insert":
                _, _, _, nj1, nj2 = opcodes[i + 1]
                inserted = " ".join(gold_tokens[nj1:nj2]).strip()
                merged = f"{tgt} {inserted}".strip()
                if src and merged:
                    spans.append((src, merged))
                i += 2
                continue
            if src and tgt:
                spans.append((src, tgt))
            i += 1
            continue

        # delete / dangling insert -> not emitted
        i += 1

    return spans


def make_span_target(stt: str, gold: str) -> str:
    """Build the span-level target string ``S`` (Sec. 4.2.1)."""
    spans = extract_spans(stt, gold)
    parts = [f"{a} {ARROW_TOKEN} {b}".strip() for a, b in spans if a.strip() and b.strip()]
    if not parts:
        return NO_ERROR
    return f" {SEP_TOKEN} ".join(parts).strip()


def apply_spans(utterance: str, span_string: str) -> str:
    """Reconstruct the corrected utterance from a predicted span string (Sec. 4.2.3).

    Every ``span_U`` is substituted by its ``span_G`` counterpart. Two properties
    follow from the replace-only span format and are intentional (they match the
    formulation evaluated in the paper):

    * **deletions are never emitted**, so an utterance whose only error is a
      spurious extra word cannot be repaired (0.19 % of the erroneous training
      utterances);
    * substitution is textual, so if ``span_U`` occurs several times in the
      utterance every occurrence is rewritten.

    Applying gold span targets recovers the ground truth for 98.9 % of the
    erroneous training utterances, which is the ceiling of this design.
    """
    span_string = str(span_string).strip()
    if not span_string or span_string == NO_ERROR:
        return str(utterance)

    text = str(utterance)
    for segment in span_string.split(SEP_TOKEN):
        segment = segment.strip()
        if not segment or ARROW_TOKEN not in segment:
            continue
        src, tgt = segment.split(ARROW_TOKEN, 1)
        src, tgt = src.strip(), tgt.strip()
        if src and tgt:
            text = text.replace(src, tgt)
    return text


def word_wer(ref: str, hyp: str) -> float:
    """Word-level WER: Levenshtein distance normalised by ``len(ref.split())``."""
    ref_words = str(ref).split()
    hyp_words = str(hyp).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(ref_words) + 1)
    dp[0, :] = np.arange(len(hyp_words) + 1)
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[len(ref_words), len(hyp_words)]) / len(ref_words)


def count_spans(span_string: str) -> int:
    """Number of error spans encoded in a span string (0 for ``이상없음``)."""
    span_string = str(span_string).strip()
    if not span_string or span_string == NO_ERROR:
        return 0
    return span_string.count(ARROW_TOKEN)
