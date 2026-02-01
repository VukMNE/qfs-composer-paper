# QAGS II - Computation of BERTScore

#Computes BERTScore metric inside QAGS by comparing answers produced from QA when
#  it is conditioned only on summary and when it is conditioned only on source document.

# !pip install -U bitsandbytes
# !pip install transformers
# !pip uninstall -y datasets
# !pip install -U "datasets==2.20.0" "accelerate>=0.33.0" "huggingface-hub>=0.24.0"
# # (optional but helpful)
# !pip install -U "evaluate>=0.4.2"
# !pip -q install bert-score

# 1) Imports
import os, math, collections
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable
import json
from pathlib import Path
from statistics import mean, pstdev
from bert_score import score as bertscore




import numpy as np
import evaluate
import torch

from datasets import load_dataset, Features, Value, Sequence, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)


# ========= CONFIG =========
# Point this to your file on Drive or upload via the Colab file browser and set the path here.
INPUT_PATH = "/content/drive/MyDrive/qags_results/"      # <-- change this
SAVE_AUGMENTED_JSON = True                     # set to False if you don't want to save
OUTPUT_PATH = "/content/qags_scores_"

# BERTScore model config (multilingual; good for Slovenian)
MODEL_TYPE = "xlm-roberta-large"               # consider "xlm-roberta-base" if low memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16                                # adjust if you hit OOM

RESCALE_WITH_BASELINE = False
LANG = "sl"


from tqdm import tqdm

# ========= HELPERS =========
def load_examples(path: str) -> List[Dict[str, Any]]:
    """Load evaluation examples from JSON or JSONL with a few robust fallbacks."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Try standard JSON first
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # might be a single example or a container field
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            if "examples" in data and isinstance(data["examples"], list):
                return data["examples"]
            # assume it's a single-example file
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: JSON Lines
    examples = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                examples.append(ex)
            except json.JSONDecodeError:
                # skip bad lines
                continue
    if not examples:
        raise ValueError("Could not parse the input file as JSON or JSONL.")
    return examples


def is_empty_text(s: Any) -> bool:
    return not isinstance(s, str) or (s.strip() == "")


def compute_bertscore_for_pairs(
    candidates: List[str], references: List[str],
    mdl_name, setting_name
) -> List[float]:
    """
    Compute BERTScore F1 for (candidate, reference) pairs.
    Returns list of floats (F1), same length as inputs.
    Empty/invalid pairs return 0.0.
    """
    assert len(candidates) == len(references)
    n = len(candidates)

    # Identify valid (non-empty) pairs; BERTScore throws on empties.
    valid_idx = []
    valid_cands = []
    valid_refs = []

    for i, (c, r) in tqdm(enumerate(zip(candidates, references)), desc=f"Evaluating BERTSCORE for {mdl_name} : {setting_name}"):
        if is_empty_text(c) or is_empty_text(r):
            continue
        valid_idx.append(i)
        valid_cands.append(c.strip())
        valid_refs.append(r.strip())

    f1_scores = [0.0] * n  # default 0.0 for invalid pairs

    if valid_cands:
        # Compute in batches to control memory usage
        start = 0
        while start < len(valid_cands):
            end = min(start + BATCH_SIZE, len(valid_cands))
            batch_cands = valid_cands[start:end]
            batch_refs = valid_refs[start:end]

            P, R, F1 = bertscore(
                batch_cands,
                batch_refs,
                model_type=MODEL_TYPE,
                device=DEVICE,
                lang=LANG if RESCALE_WITH_BASELINE else None,
                rescale_with_baseline=RESCALE_WITH_BASELINE,
                batch_size=min(BATCH_SIZE, len(batch_cands)),
                verbose=False,
            )
            # Assign back
            for j, f1 in enumerate(F1):
                f1_scores[valid_idx[start + j]] = float(f1.item())

            start = end

    return f1_scores


def as_number(x) -> float:
    """Return float if x is numeric-like, else None."""
    try:
        if isinstance(x, bool):  # avoid True/False being treated as 1/0 accidentally
            return float(int(x))
        return float(x)
    except (TypeError, ValueError):
        return None


def collect_metric(values: Iterable[Any]) -> List[float]:
    """Filter to numeric floats; ignore missing/non-numeric."""
    out = []
    for v in values:
        fv = as_number(v)
        if fv is not None and not (isinstance(fv, float) and math.isnan(fv)):
            out.append(float(fv))
    return out


def safe_mean(xs: List[float]) -> float:
    return mean(xs) if xs else float("nan")


def safe_pstdev(xs: List[float]) -> float:
    return pstdev(xs) if len(xs) > 1 else 0.0 if len(xs) == 1 else float("nan")

# ========= MAIN COMPUTATION FUNCTION =========

