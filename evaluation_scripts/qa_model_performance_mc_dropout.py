# !pip install -U bitsandbytes
# !pip install transformers
# !pip uninstall -y datasets
# !pip install -U "datasets==2.20.0" "accelerate>=0.33.0" "huggingface-hub>=0.24.0"
# # (optional but helpful)
# !pip install -U "evaluate>=0.4.2"
# !pip -q install bert-score


from bert_score import score as bertscore

# 1) Imports
import os, math, collections
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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

# 2) Paths to your JSON files (adjust as needed)
#    Your file is SQuAD-v2 style with a top-level "data" field.
DATA_FILES = {
    # If you only have one file now, you can point both to the same path.
    "train": "/content/drive/MyDrive/squad2-slo-mt-train.json",
    "validation": "/content/drive/MyDrive/squad2-slo-mt-dev.json",
}

# 3) Define the schema explicitly (prevents Arrow casting issues)
features = Features({
    "id": Value("string"),
    "title": Value("string"),
    "context": Value("string"),
    "question": Value("string"),
    "answers": {
        "text": Sequence(Value("string")),
        "answer_start": Sequence(Value("int64")),
    },
})



# import json
# train_ds = dataset["train"]
# train_ds = train_ds.select(range(1000, 1100))
# train_ds


# # To save as a single, standard JSON file, set lines=False
# train_ds.to_json("squad_dataset_mini.json", lines=False)

# 4) Load the dataset (note field="data")

DATA_FILES = {
    # If you only have one file now, you can point both to the same path.
    "test": "/content/squad_slo_mini_test.json",
}

test_dataset: DatasetDict = load_dataset(
    "json",
    data_files=DATA_FILES,
    field="data",
    features=features,
)

print(test_dataset)




# ===========================================
# Full-model MC-Dropout QA evaluation (GaMS-9B-Instruct, no LoRA)
# ===========================================
import os, math, re, json, random, textwrap, sys
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# -----------------------------
# Paths / settings
# -----------------------------
CKPT_DIR = "VukDju/GaMS-9B-Instruct-QA-3ep"  # <-- your saved model dir from training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123

# Eval subset (to keep time reasonable)
EVAL_LIMIT = None           # set None to use full validation set
K_MC = 20                  # MC passes per example

# Decoding
MAX_NEW_TOKENS = 24
DO_SAMPLE = False          # keep greedy; dropout is the only randomness
TEMPERATURE = 1.0
TOP_P = 1.0

# Choose a multilingual checkpoint that handles Slovene well.
BERTSCORE_MODEL = "xlm-roberta-large"   # fallback: "xlm-roberta-base"
BERTSCORE_BATCH_SIZE = 32               # lower if you see RAM issues
BERTSCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def first_gold(answers_dict):
    if not isinstance(answers_dict, dict):
        return ""
    texts = answers_dict.get("text", [])
    if isinstance(texts, list) and len(texts) > 0:
        return str(texts[0])
    if isinstance(texts, str):
        return texts
    return ""

def bertscore_for_mc(gold: str, mc_samples: List[str]) -> Dict[str, Any]:
    """
    Returns:
      - f1_list: list[float] of BERTScore F1 per MC sample
      - mean, std, iqr, best, worst, median
      - ci_low, ci_high (bootstrap 95% CI of the mean, small K-safe)
    """
    refs = [gold for _ in mc_samples]
    # bertscore returns P,R,F1 tensors in the same order as cands
    _, _, F1 = bertscore(
        cands=mc_samples,
        refs=refs,
        model_type=BERTSCORE_MODEL,
        lang="sl",                         # Slovene
        rescale_with_baseline=True,        # recommended for comparability
        batch_size=BERTSCORE_BATCH_SIZE,
        device=BERTSCORE_DEVICE,
        verbose=False,
    )
    f1 = F1.detach().cpu().numpy().astype(float)
    if len(f1) == 0:
        return {"f1_list": [], "mean": 0.0, "std": 0.0, "iqr": 0.0, "best": 0.0, "worst": 0.0,
                "median": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    f1_sorted = np.sort(f1)
    q25, q75 = np.percentile(f1_sorted, 25), np.percentile(f1_sorted, 75)
    # tiny bootstrap for a CI on the mean (works fine for K≈20)
    rng = np.random.default_rng(12345)
    boots = []
    for _ in range(200):
        samp = rng.choice(f1, size=len(f1), replace=True)
        boots.append(samp.mean())
    boots = np.sort(boots)
    ci_low, ci_high = boots[int(0.025*len(boots))], boots[int(0.975*len(boots))]

    return {
        "f1_list": f1.tolist(),
        "mean": float(f1.mean()),
        "std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        "iqr": float(q75 - q25),
        "best": float(f1.max()),
        "worst": float(f1.min()),
        "median": float(np.median(f1)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }

# Output verbosity
DEBUG_PRINT = True        # True = print per-example details
DEBUG_JSONL = "/content/qa_mc_debug.jsonl"

# -----------------------------
# Repro
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Load dataset
# -----------------------------
val_ds = test_dataset["test"]
print(f"Validation examples: {len(val_ds)}")

# -----------------------------
# Tokenizer & Model (force eager attention so dropout is honored)
# -----------------------------
tok = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token



# If your training updated dropout via config, it’s already saved inside CKPT_DIR.
# We still force 'eager' attention to ensure attention-dropout isn't bypassed.
model = AutoModelForCausalLM.from_pretrained(
    CKPT_DIR,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()


# -----------------------------
# Utilities
# -----------------------------
def shorten_ctx(s: str, n: int = 500) -> str:
    return textwrap.shorten(s.replace("\n", " "), width=n, placeholder=" ...")

ROLE_PREFIX_RE = re.compile(r"^\s*(model|assistant|asistent|assistantu)?\s*[:\-]?\s*", re.IGNORECASE)
END_TOK_RE = re.compile(r"\[?\s*END\s*\]?", re.IGNORECASE)

NO_ANS_TOKEN = "<no_answer>"

def clean_answer_text(s: str) -> str:
    if "Odgovor:" in s:
        s = s.split("Odgovor:")[-1]
    s = END_TOK_RE.sub("", s)
    s = ROLE_PREFIX_RE.sub("", s)
    s = s.replace("\u200b", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()

def normalize_answer(s: str) -> str:
    s = clean_answer_text(s).lower()
    # map literal no-answer token to a canonical marker
    if NO_ANS_TOKEN in s.split():
        return NO_ANS_TOKEN
    table = str.maketrans({c: " " for c in ",.;:!?()[]{}\"'`"})
    s = s.translate(table)
    s = " ".join(s.split())
    # if it's now empty, keep it empty (not the same as NO_ANS_TOKEN)
    return s

def is_no_answer(s: str) -> bool:
    return normalize_answer(s) == NO_ANS_TOKEN

def exact_match(pred: str, gold: str) -> int:
    npred = normalize_answer(pred)
    ngold = normalize_answer(gold)
    # Treat no-answer as its own class
    return 1 if npred == ngold else 0

def f1_score(pred: str, gold: str) -> float:
    # If both abstain, count as perfect
    if is_no_answer(pred) and is_no_answer(gold):
        return 1.0
    # If only one abstains, it's a miss
    if is_no_answer(pred) != is_no_answer(gold):
        return 0.0
    # Else, token F1 on text
    pt = [t for t in normalize_answer(pred).split() if t != NO_ANS_TOKEN]
    gt = [t for t in normalize_answer(gold).split() if t != NO_ANS_TOKEN]
    if not pt and not gt: return 1.0
    if not pt or not gt:  return 0.0
    common = {}
    for t in pt: common[t] = min(pt.count(t), gt.count(t))
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    prec = num_same / len(pt)
    rec  = num_same / len(gt)
    return 2 * prec * rec / (prec + rec)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins-1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask): continue
        ece += (mask.sum()/len(probs)) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)

def optimize_temperature(z: np.ndarray, y: np.ndarray) -> float:
    best_T, best_loss = 1.0, 1e9
    for T in np.logspace(-1, 1.2, 40):  # 0.1 .. ~15.8
        p = sigmoid(z / T)
        eps = 1e-12
        loss = -(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)).mean()
        if loss < best_loss:
            best_loss, best_T = loss, float(T)
    return best_T

# -----------------------------
# Prompt builder (same as training)
# -----------------------------
def build_qa_prompt(question: str, context: str) -> str:
    user = (
        "Na podlagi podanega vprašanja in besedila generiraj samo en smiseln in pravilen odgovor, "
        "ki izhaja izključno iz konteksta tega besedila. "
        "Odgovor naj bo jasen, natančen in kratek (le nekaj besed). "
        "Če na vprašanje ni mogoče odgovoriti na podlagi predloženega besedila, ne ustvarite nobenega besedila."
        f"\n\nVprašanje: {question}\nBesedilo: {context}\nOdgovor:"
    )
    messages = [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_qa_prompt_plain(question: str, context: str) -> str:
    return (
        "Na podlagi podanega vprašanja in besedila generiraj samo en smiseln in pravilen odgovor, "
        "ki izhaja izključno iz konteksta tega besedila. "
        "Odgovor naj bo jasen, natančen in kratek (le nekaj besed). "
        f"Če na vprašanje ni mogoče odgovoriti na podlagi predloženega besedila, vrni {NO_ANS_TOKEN}."
        f"\nVprašanje: {question}\nBesedilo: {context}\nOdgovor:"
    )

# -----------------------------
# Generation (returns text, mean token logprob, mean token entropy)
# -----------------------------

END_ID = tok.convert_tokens_to_ids("[END]")

# UNCOMMENT FOR 9B
def generate_once(prompt: str) -> Tuple[str, float, float]:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            eos_token_id=END_ID,        # hard stop
            top_p=TOP_P,
            use_cache=False,                 # critical for MC
            return_dict_in_generate=True,
            output_scores=True,
        )
    text = tok.decode(out.sequences[0], skip_special_tokens=True)
    # compute mean token logprob & entropy
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    scores = out.scores
    lps, ents = [], []
    for step_logits, tid in zip(scores, gen_ids):
        logp = step_logits.log_softmax(dim=-1)[0, tid.item()].item()
        lps.append(logp)
        probs = step_logits.softmax(dim=-1)[0]
        ents.append(float(-(probs * probs.log()).sum().item()))
    mean_lp = float(np.mean(lps)) if lps else -1e9
    mean_ent = float(np.mean(ents)) if ents else 0.0
    return text, mean_lp, mean_ent

# def generate_once(prompt_str: str) -> tuple[str, float, float]:
#     # IMPORTANT: no chat template here; we pass the raw text,
#     # and we do *not* add special tokens (to match training)
#     inputs = tok(prompt_str, return_tensors="pt", add_special_tokens=False).to(model.device)

#     with torch.no_grad():
#         out = model.generate(
#             **inputs,
#             max_new_tokens=24,
#             min_new_tokens=1,              # diagnostic; remove once you see non-empty outputs
#             do_sample=False,               # MC randomness comes only from dropout
#             temperature=1.0,
#             top_p=1.0,
#             use_cache=False,               # critical for MC-Dropout
#             return_dict_in_generate=True,
#             output_scores=True,
#             # don't set eos_token_id to any custom token unless you've added it to the tokenizer
#             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
#         )

#     # decode and compute mean token logprob / entropy (as before)
#     text = tok.decode(out.sequences[0], skip_special_tokens=True)
#     gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
#     lps, ents = [], []
#     for step_logits, tid in zip(out.scores, gen_ids):
#         logp = step_logits.log_softmax(dim=-1)[0, tid.item()].item()
#         lps.append(logp)
#         probs = step_logits.softmax(dim=-1)[0]
#         ents.append(float(-(probs * probs.log()).sum().item()))
#     mean_lp = float(np.mean(lps)) if lps else -1e9
#     mean_ent = float(np.mean(ents)) if ents else 0.0
#     return text, mean_lp, mean_ent


# -----------------------------
# MC-Dropout runner over a single prompt
# -----------------------------
def mc_dropout_predict(prompt: str, k: int = K_MC) -> Dict[str, Any]:
    samples, confs, ents = [], [], []
    # Enable dropout across the full model
    model.train()
    with torch.no_grad():
        for _ in range(k):
            s, lp, ent = generate_once(prompt)
            s_clean = clean_answer_text(s)
            samples.append(s_clean)
            confs.append(lp)
            ents.append(ent)
            print(f"    MC[{_+1:02d}] -> '{s_clean}' (mean_lp={lp:.3f}, entropy={ent:.3f})")
    model.eval()
    collapsed = {}
    for a in samples:
        collapsed[a] = collapsed.get(a, 0) + 1
    return {"samples": samples, "mean_logprob": confs, "mean_entropy": ents, "collapsed": collapsed}

def predictive_entropy(collapsed: Dict[str, int]) -> float:
    n = sum(collapsed.values())
    if n == 0: return 0.0
    H = 0.0
    for c in collapsed.values():
        p = c / n
        if p > 0: H -= p * math.log(p + 1e-12)
    return H

def variation_ratio(collapsed: Dict[str, int]) -> float:
    n = sum(collapsed.values())
    if n == 0: return 0.0
    return 1.0 - max(collapsed.values())/n

# -----------------------------
# Evaluate
# -----------------------------
if DEBUG_PRINT:
    # reset debug file
    with open(DEBUG_JSONL, "w", encoding="utf-8") as _f: pass

gold_texts, preds = [], []
correct_flags, conf_meanlogprob = [], []
unc_entropy, unc_varratio = [], []
no_answer_flags, is_answerable = [], []
# for bert score
bs_mean_list, bs_std_list, bs_best_list = [], [], []
sem_correct_flags = []


BS_THRESHOLD = 0.75  # if BERTScore is greater than 0.75, we will consider answer to be correct in coverage-accuracy analysis



pbar = tqdm(val_ds, desc="Evaluating QA with MC-Dropout")
for i, ex in enumerate(pbar):
    q = ex["question"]
    ctx = ex["context"]
    gold = ex["answers"]["text"] if isinstance(ex["answers"]["text"], str) else (
        ex["answers"]["text"][0] if ex["answers"]["text"] else ""
    )

    print("\n" + "="*80)
    print(f"[{i+1}] Question : {q}")
    print(f"Context : {shorten_ctx(ctx)}")
    print(f"Gold    : {gold}")
    prompt = build_qa_prompt(q, ctx) # UNCOMMENT for 9B
    #prompt = build_qa_prompt_plain(q, ctx)


    # MC-Dropout passes
    mc = mc_dropout_predict(prompt, K_MC)
    collapsed = mc["collapsed"]
    modal_pred = max(collapsed.items(), key=lambda kv: kv[1])[0] if collapsed else ""
    mean_lp = float(np.mean(mc["mean_logprob"])) if mc["mean_logprob"] else -1e9
    print(f"Modal prediction: '{modal_pred}' (top answer across {K_MC} MC passes)")

    # --- BERTScore over MC samples (Slovene) ---
    bs = bertscore_for_mc(gold, mc["samples"])
    # Save optional per-example diagnostics
    if DEBUG_PRINT:
        print(f"BERTScore F1 (MC): mean={bs['mean']:.3f}±{bs['std']:.3f}  "
              f"[{bs['ci_low']:.3f}, {bs['ci_high']:.3f}]  "
              f"best={bs['best']:.3f}  median={bs['median']:.3f}  iqr={bs['iqr']:.3f}")

    # record
    gold_texts.append(gold)
    preds.append(modal_pred)
    conf_meanlogprob.append(mean_lp)
    H = predictive_entropy(collapsed)
    V = variation_ratio(collapsed)
    unc_entropy.append(H)
    unc_varratio.append(V)
    bs_mean_list.append(bs["mean"])
    bs_std_list.append(bs["std"])
    bs_best_list.append(bs["best"])
    sem_correct = 1 if bs["mean"] >= BS_THRESHOLD else 0
    sem_correct_flags.append(sem_correct)



    gold_na_flag = 1 if normalize_answer(gold) == NO_ANS_TOKEN or gold.strip() == "" else 0
    pred_na_flag = 1 if is_no_answer(modal_pred) else 0
    is_answerable.append(1 - gold_na_flag)
    no_answer_flags.append(pred_na_flag)
    correct_flags.append(exact_match(modal_pred, gold))

    if DEBUG_PRINT:
        # one-line console summary
        print("\n" + "="*80)
        print(f"[#{i+1}] Q: {q}")
        print(f"Ctx: {shorten_ctx(ctx)}")
        print(f"Gold: {gold}")
        print(f"Modal: {modal_pred} | H={H:.4f} V={V:.4f} mean_lp={mean_lp:.3f}")
        print("MC top-3:", "; ".join([f"'{a}'×{c}" for a,c in sorted(collapsed.items(), key=lambda kv: kv[1], reverse=True)[:3]]))
        # JSONL dump
        with open(DEBUG_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "index": i, "question": q, "context": ctx, "gold": gold,
                "modal_pred": modal_pred, "collapsed": collapsed,
                "mc_mean_logprob": mc["mean_logprob"], "mc_mean_entropy": mc["mean_entropy"]
            }, ensure_ascii=False) + "\n")

# -----------------------------
# Metrics
# -----------------------------
def macro_em_f1(preds, golds):
    ems, f1s = [], []
    for p, g in zip(preds, golds):
        ems.append(exact_match(p, g))
        f1s.append(f1_score(p, g))
    return float(np.mean(ems)), float(np.mean(f1s))

EM, F1 = macro_em_f1(preds, gold_texts)

# Heuristic unanswerable confusion (modal empty == predict no-answer)
gold_unans = np.array([1 - a for a in is_answerable], dtype=np.int32)  # 1 == no-answer
pred_unans = np.array(no_answer_flags, dtype=np.int32)
tn = int(((gold_unans == 1) & (pred_unans == 1)).sum())
tp = int(((gold_unans == 0) & (pred_unans == 0)).sum())
fp = int(((gold_unans == 1) & (pred_unans == 0)).sum())
fn = int(((gold_unans == 0) & (pred_unans == 1)).sum())

# Calibration from mean token logprob
z = np.array(conf_meanlogprob, dtype=np.float64)
y = np.array(correct_flags, dtype=np.int32)

probs_before = 1.0/(1.0 + np.exp(-z))  # T=1
ece_before = compute_ece(probs_before, y, n_bins=15)

T_star = optimize_temperature(z, y)
probs_after = 1.0/(1.0 + np.exp(-z / T_star))
ece_after = compute_ece(probs_after, y, n_bins=15)

# Coverage–risk
order = np.argsort(-probs_after)
labels_sorted = y[order]
cov_points, acc_points = [], []
steps = list(range(10, len(labels_sorted)+1, max(1, len(labels_sorted)//10)))
for k in steps:
    cov_points.append(k/len(labels_sorted))
    acc_points.append(float(labels_sorted[:k].mean()))

# -----------------------------
# Print summary
# -----------------------------
print("\n=== QA Intrinsic Metrics ===")
print(f"Examples evaluated: {len(preds)}")
print(f"EM: {EM:.4f}")
print(f"F1: {F1:.4f}")

print("\n=== Unanswerable (heuristic modal-empty) ===")
print(f"TN: {tn} | TP: {tp} | FP: {fp} | FN: {fn}")

print("\n=== Calibration ===")
print(f"Temperature* (optimized): {T_star:.3f}")
print(f"ECE before: {ece_before:.4f}")
print(f"ECE after : {ece_after:.4f}")

print("\n=== Uncertainty (MC-Dropout, string-level) ===")
print(f"Predictive entropy (mean ± std): {np.mean(unc_entropy):.4f} ± {np.std(unc_entropy):.4f}")
print(f"Variation ratio (mean ± std):    {np.mean(unc_varratio):.4f} ± {np.std(unc_varratio):.4f}")

print("\n=== Coverage–Risk (after calibration) ===")
for c, a in zip(cov_points, acc_points):
    print(f"Coverage {c:6.1%} | Accuracy {a:6.2%}")

print('==== BERTSCORE =====')
print("\n=== BERTScore (Slovene, over MC samples) ===")
if bs_mean_list:
    print(f"Mean of per-example BERTScore means: {np.mean(bs_mean_list):.4f}")
    print(f"Mean of per-example BERTScore stds : {np.mean(bs_std_list):.4f}")
    print(f"Mean of per-example BERTScore bests: {np.mean(bs_best_list):.4f}")

    # A simple 'coverage-risk' using BERTScore mean as confidence:
    order_bs = np.argsort(-np.array(bs_mean_list))
    labels_sorted_bs = np.array(correct_flags)[order_bs]
    cov_points_bs, acc_points_bs = [], []
    steps = list(range(10, len(labels_sorted_bs)+1, max(1, len(labels_sorted_bs)//10)))
    for k in steps:
        cov_points_bs.append(k/len(labels_sorted_bs))
        acc_points_bs.append(float(labels_sorted_bs[:k].mean()))
    print("\n=== Coverage–Risk (BERTScore mean as confidence) ===")
    for c, a in zip(cov_points_bs, acc_points_bs):
        print(f"Coverage {c:6.1%} | Accuracy {a:6.2%}")


    # -----------------------------
    # Coverage–risk (semantic, BERTScore thresholded)
    # -----------------------------
    sem_flags = np.array(sem_correct_flags, dtype=np.int32)
    order_bs = np.argsort(-np.array(bs_mean_list))  # sort by mean BERTScore confidence
    labels_sorted_bs = sem_flags[order_bs]

    cov_points_bs, acc_points_bs = [], []
    steps = list(range(10, len(labels_sorted_bs)+1, max(1, len(labels_sorted_bs)//10)))
    for k in steps:
        cov_points_bs.append(k/len(labels_sorted_bs))
        acc_points_bs.append(float(labels_sorted_bs[:k].mean()))

    print("\n=== Coverage–Risk (BERTScore-thresholded, semantic) ===")
    for c, a in zip(cov_points_bs, acc_points_bs):
        print(f"Coverage {c:6.1%} | Accuracy {a:6.2%}")
else:
    print("No BERTScore entries computed.")







# ===========================================
# Summarize QA MC-Dropout with Unanswerable-Aware BERTScore + Thresholded Semantic Accuracy (Colab)


# @title Summarize QA MC-Dropout with Unanswerable-Aware BERTScore + Thresholded Semantic Accuracy (Colab) { display-mode: "form" }
# @markdown **Path to your debug file (JSONL):**
PATH = "/content/qa_mc_debug.jsonl"  # @param {type:"string"}
# @markdown **BERTScore settings (used only when both gold & cand are non-empty):**
BERTSCORE_MODEL = "xlm-roberta-large"  # @param {type:"string"}
BERTSCORE_LANG = "sl"                  # @param {type:"string"}
BERTSCORE_RESCALE = False              # @param {type:"boolean"}
BERTSCORE_IDF = True                   # @param {type:"boolean"}
BERTSCORE_BATCH_SIZE = 32              # @param {type:"number"}
# @markdown **Threshold for semantic correctness (BERTScore F1):**
BS_THRESHOLD = 0.80                    # @param {type:"number"}

import sys, subprocess, os, json, re, math, time
from collections import Counter
from typing import Dict, Any, List
def pipi(pkg): subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# deps
try:
    from bert_score import score as bertscore
except ImportError:
    pipi("bert-score>=0.3.13"); from bert_score import score as bertscore
try:
    from tqdm.auto import tqdm
except Exception:
    pipi("tqdm>=4.66.0"); from tqdm.auto import tqdm

import numpy as np
import torch

NO_ANS_TOKEN = "<no_answer>"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- text utils ----------
ROLE_PREFIX_RE = re.compile(r"^\s*(model|assistant|asistent|assistantu|asistentu)?\s*[:\-]?\s*", re.IGNORECASE)
END_TOK_RE     = re.compile(r"\[?\s*END\s*\]?", re.IGNORECASE)

def clean_answer_text(s: str) -> str:
    if not s: return ""
    if "Odgovor:" in s: s = s.split("Odgovor:")[-1]
    s = END_TOK_RE.sub("", s)
    s = ROLE_PREFIX_RE.sub("", s)
    s = s.replace("\u200b"," ").replace("\r"," ").replace("\n"," ")
    return " ".join(s.split())

def normalize_answer(s: str) -> str:
    s = clean_answer_text(s).lower()
    if NO_ANS_TOKEN in s.split(): return NO_ANS_TOKEN
    table = str.maketrans({c:" " for c in ",.;:!?()[]{}\"'`"})
    return " ".join(s.translate(table).split())

def is_no_ans_pred(s: str) -> bool:
    ns = normalize_answer(s)
    return ns in ("", NO_ANS_TOKEN)

def is_no_ans_gold(s: str) -> bool:
    ns = normalize_answer(s)
    return ns in ("", NO_ANS_TOKEN)

# ---------- EM / F1 (unanswerable-aware) ----------
def exact_match(pred: str, gold: str) -> int:
    npred, ngold = normalize_answer(pred), normalize_answer(gold)
    if npred in ("", NO_ANS_TOKEN) and ngold in ("", NO_ANS_TOKEN):
        return 1
    return int(npred == ngold)

def f1_score(pred: str, gold: str) -> float:
    p_no, g_no = is_no_ans_pred(pred), is_no_ans_gold(gold)
    if p_no and g_no: return 1.0
    if p_no != g_no:  return 0.0
    pt = [t for t in normalize_answer(pred).split() if t != NO_ANS_TOKEN]
    gt = [t for t in normalize_answer(gold).split() if t != NO_ANS_TOKEN]
    if not pt and not gt: return 1.0
    if not pt or not gt:  return 0.0
    common = Counter(pt) & Counter(gt)
    same = sum(common.values());
    if same == 0: return 0.0
    prec, rec = same/len(pt), same/len(gt)
    return 2*prec*rec/(prec+rec)

# ---------- MC helpers ----------
def predictive_entropy(collapsed: Dict[str,int]) -> float:
    n = sum(collapsed.values())
    if n <= 0: return 0.0
    H = 0.0
    for c in collapsed.values():
        p = c/n
        if p > 0: H -= p*math.log(p+1e-12)
    return H

def variation_ratio(collapsed: Dict[str,int]) -> float:
    n = sum(collapsed.values())
    if n <= 0: return 0.0
    return 1.0 - max(collapsed.values())/n

# ---------- calibration ----------
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def compute_ece(probs, labels, n_bins=15):
    bins = np.linspace(0,1,n_bins+1); ece=0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs>=lo) & (probs<(hi if i<n_bins-1 else hi+1e-9))
        if not np.any(mask): continue
        ece += (mask.sum()/len(probs))*abs(labels[mask].mean()-probs[mask].mean())
    return float(ece)

def optimize_temperature(z,y):
    best_T, best_loss = 1.0, 1e9
    for T in np.logspace(-1,1.2,40):
        p = sigmoid(z/T)
        loss = -(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)).mean()
        if loss < best_loss: best_loss, best_T = loss, float(T)
    return best_T

# ---------- I/O ----------
def ensure_file(path: str) -> str:
    if os.path.exists(path): return path
    try:
        from google.colab import files
        print(f"[I/O] {path} not found. Please upload qa_mc_debug.jsonl …")
        uploaded = files.upload()
        if not uploaded: raise FileNotFoundError("No file uploaded.")
        fname = list(uploaded.keys())[0]
        print(f"[I/O] Using uploaded: {fname}")
        return fname
    except Exception as e:
        raise FileNotFoundError(f"Cannot open {path}: {e}")

def load_jsonl(path: str) -> List[Dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in tqdm(f, desc="[Load] Lines", unit="ln"):
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: pass
    return rows

def reconstruct_samples(collapsed: Dict[str,int]) -> List[str]:
    out=[]
    for a,c in (collapsed or {}).items():
        out.extend([a]*int(c))
    return out

# ---------- Unanswerable-aware BERTScore for many pairs ----------
def compute_bs_pairs_unans_aware(cands: List[str], refs: List[str]) -> np.ndarray:
    """
    Rules:
      - gold empty/no_answer & cand no_answer -> 1.0
      - gold empty/no_answer & cand non-empty -> 0.0
      - gold non-empty & cand no_answer       -> 0.0
      - else compute real BERTScore
    """
    assert len(cands)==len(refs)
    N = len(cands)
    out = np.empty(N, dtype=np.float64)

    ref_no  = np.array([is_no_ans_gold(r) for r in refs], dtype=bool)
    cand_no = np.array([is_no_ans_pred(c) for c in cands], dtype=bool)

    both_no_mask          = ref_no & cand_no
    ref_no_cand_yes_mask  = ref_no & (~cand_no)
    ref_yes_cand_no_mask  = (~ref_no) & cand_no

    out[both_no_mask]         = 1.0
    out[ref_no_cand_yes_mask] = 0.0
    out[ref_yes_cand_no_mask] = 0.0

    needs_bs_mask = ~(both_no_mask | ref_no_cand_yes_mask | ref_yes_cand_no_mask)
    idx = np.nonzero(needs_bs_mask)[0]
    if len(idx) > 0:
        bs_batch = int(BERTSCORE_BATCH_SIZE)
        for i in tqdm(range(0, len(idx), bs_batch), desc="[Stage 3] BERTScore batches", unit="batch"):
            sl = idx[i:i+bs_batch]
            c_batch = [cands[j] for j in sl]
            r_batch = [refs[j]  for j in sl]
            _,_,F1 = bertscore(
                cands=c_batch, refs=r_batch,
                model_type=BERTSCORE_MODEL, lang=BERTSCORE_LANG,
                rescale_with_baseline=BERTSCORE_RESCALE, idf=BERTSCORE_IDF,
                batch_size=bs_batch, device=DEVICE, verbose=False
            )
            out[sl] = F1.detach().cpu().numpy().astype("float64")
    return out

# =================== PIPELINE ===================
t_total0 = time.perf_counter()
PATH = ensure_file(PATH)

print("[Start] Loading JSONL …")
rows = load_jsonl(PATH)
if not rows: raise RuntimeError("Empty file.")

N = len(rows)
print(f"[Info] Total examples: {N}")

# ---- Stage 1: EM/F1, flags, MC uncertainty ----
print("[Stage 1] Metrics & MC stats …")
t0 = time.perf_counter()
ems,f1s,correct,zs,is_ans,pred_na,unc_H,unc_V=[],[],[],[],[],[],[],[]
for ex in tqdm(rows, desc="[Stage 1] Examples", unit="ex"):
    gold, modal = ex.get("gold",""), ex.get("modal_pred","")
    em = exact_match(modal,gold); ems.append(em); correct.append(em)
    f1s.append(f1_score(modal,gold))
    is_ans.append(0 if is_no_ans_gold(gold) else 1)
    pred_na.append(1 if is_no_ans_pred(modal) else 0)

    coll = ex.get("collapsed",{}) or {}
    n = sum(coll.values())
    if n>0:
        pmax = max(coll.values())/n
        unc_V.append(1-pmax)
        H=0.0
        for c in coll.values():
            p=c/n;  H -= p*math.log(p+1e-12)
        unc_H.append(H)
    else:
        unc_V.append(0.0); unc_H.append(0.0)

    lps = ex.get("mc_mean_logprob",[]) or []
    zs.append(np.mean(lps) if lps else -1e9)
print(f"[Stage 1] Done in {time.perf_counter()-t0:.2f}s")

# ---- Stage 2: Build BERTScore pairs (all examples, all MC samples) ----
print("[Stage 2] Building (candidate, reference) pairs from MC samples …")
t0 = time.perf_counter()
cand_flat, ref_flat, ex_slices = [], [], []  # ex_slices maps per-example segment
for ex in tqdm(rows, desc="[Stage 2] Examples", unit="ex"):
    gold = ex.get("gold","")
    samples = reconstruct_samples(ex.get("collapsed",{}))
    start = len(cand_flat)
    if samples:
        cand_flat.extend(samples)
        ref_flat.extend([gold]*len(samples))
    end = len(cand_flat)
    ex_slices.append((start,end))
print(f"[Stage 2] Collected {len(cand_flat)} pairs in {time.perf_counter()-t0:.2f}s")

# ---- Stage 3: Compute unanswerable-aware BERTScore for all pairs ----
bs_f1_flat = np.array([], dtype=np.float64)
if len(cand_flat)>0:
    print("[Stage 3] Computing unanswerable-aware BERTScore …")
    t0 = time.perf_counter()
    bs_f1_flat = compute_bs_pairs_unans_aware(cand_flat, ref_flat)
    print(f"[Stage 3] Done in {time.perf_counter()-t0:.2f}s")

# ---- Stage 4: Aggregate per example ----
print("[Stage 4] Aggregating per-example BERTScore stats …")
t0 = time.perf_counter()
bs_mean_list, bs_std_list, bs_best_list = [], [], []
bs_mean_per_ex = [None]*N
for i,(s,e) in enumerate(tqdm(ex_slices, desc="[Stage 4] Examples", unit="ex")):
    if e<=s: bs_mean_per_ex[i]=None; continue
    seg = bs_f1_flat[s:e]
    bs_mean_list.append(float(seg.mean()))
    bs_std_list.append(float(seg.std(ddof=1)) if len(seg)>1 else 0.0)
    bs_best_list.append(float(seg.max()))
    bs_mean_per_ex[i] = float(seg.mean())
print(f"[Stage 4] Done in {time.perf_counter()-t0:.2f}s")

# ---- Final metrics (EM/F1 & unanswerable) ----
EM, F1m = float(np.mean(ems)), float(np.mean(f1s))
gold_unans = np.array([1-a for a in is_ans], dtype=np.int32)
pred_unans = np.array(pred_na, dtype=np.int32)
tn = int(((gold_unans==1)&(pred_unans==1)).sum())
tp = int(((gold_unans==0)&(pred_unans==0)).sum())
fp = int(((gold_unans==1)&(pred_unans==0)).sum())
fn = int(((gold_unans==0)&(pred_unans==1)).sum())

# ---- Calibration on mean logprobs (EM labels) ----
z = np.array([v for v in zs if v>-1e8], dtype=np.float64)
y = np.array([lab for v,lab in zip(zs,ems) if v>-1e8], dtype=np.int32)
if len(z)>0:
    probs_before = sigmoid(z); ece_before = compute_ece(probs_before,y)
    T_star = optimize_temperature(z,y)
    probs_after = sigmoid(z/T_star); ece_after = compute_ece(probs_after,y)
else:
    T_star = float("nan"); ece_before = ece_after = float("nan"); probs_after = np.array([]); y = np.array([])

# ---- Coverage–risk (EM accuracy, after calibration) ----
cov_points, acc_points = [], []
if len(probs_after)>0:
    order = np.argsort(-probs_after); labels_sorted = y[order]
    steps = list(range(10, len(labels_sorted)+1, max(1, len(labels_sorted)//10)))
    for k in steps:
        cov_points.append(k/len(labels_sorted))
        acc_points.append(float(labels_sorted[:k].mean()))

# ---- BERTScore aggregates (over MC samples) ----
bs_mean = float(np.mean(bs_mean_list)) if bs_mean_list else float("nan")
bs_std  = float(np.mean(bs_std_list))  if bs_std_list  else float("nan")
bs_best = float(np.mean(bs_best_list)) if bs_best_list else float("nan")

# ---- Coverage–risk (BERTScore mean as confidence, EM labels) ----
cov_points_bs, acc_points_bs = [], []
if any(m is not None for m in bs_mean_per_ex):
    confs, labels = [], []
    for ex, m in zip(rows, bs_mean_per_ex):
        if m is None: continue
        confs.append(m)
        labels.append(exact_match(ex.get("modal_pred",""), ex.get("gold","")))
    confs = np.array(confs, dtype=np.float64)
    labels = np.array(labels, dtype=np.int32)
    order_bs = np.argsort(-confs); labels_sorted_bs = labels[order_bs]
    steps = list(range(10, len(labels_sorted_bs)+1, max(1, len(labels_sorted_bs)//10)))
    for k in steps:
        cov_points_bs.append(k/len(labels_sorted_bs))
        acc_points_bs.append(float(labels_sorted_bs[:k].mean()))

# ================== NEW: Thresholded semantic accuracy (BERTScore ≥ BS_THRESHOLD) ==================
sem_flags, sem_confs = [], []  # flags=correct/incorrect under semantic rule; confs=mean BERTScore
for m in bs_mean_per_ex:
    if m is None:
        continue
    sem_flags.append(1 if m >= BS_THRESHOLD else 0)
    sem_confs.append(m)

sem_acc_overall = float(np.mean(sem_flags)) if sem_flags else float("nan")

# Coverage–risk (Semantic): sort by mean BERTScore (confidence) and compute accuracy over thresholded flags
cov_points_sem, acc_points_sem = [], []
if sem_flags:
    sem_confs = np.array(sem_confs, dtype=np.float64)
    sem_flags = np.array(sem_flags, dtype=np.int32)
    order_sem = np.argsort(-sem_confs)
    flags_sorted_sem = sem_flags[order_sem]
    steps_sem = list(range(10, len(flags_sorted_sem)+1, max(1, len(flags_sorted_sem)//10)))
    for k in steps_sem:
        cov_points_sem.append(k/len(flags_sorted_sem))
        acc_points_sem.append(float(flags_sorted_sem[:k].mean()))
# ===================================================================================================

# ---- Print report ----
print("\n=== QA Intrinsic Metrics (unanswerable-aware) ===")
print(f"Examples evaluated: {N}")
print(f"EM: {EM:.4f}")
print(f"F1: {F1m:.4f}")

print("\n=== Unanswerable (explicit/empty) ===")
print(f"TN: {tn} | TP: {tp} | FP: {fp} | FN: {fn}")

print("\n=== Calibration ===")
print(f"Temperature* (optimized): {T_star:.3f}" if np.isfinite(T_star) else "Temperature*: n/a")
print(f"ECE before: {ece_before:.4f}" if np.isfinite(ece_before) else "ECE before: n/a")
print(f"ECE after : {ece_after:.4f}"  if np.isfinite(ece_after)  else "ECE after : n/a")

print("\n=== Uncertainty (MC-Dropout) ===")
print(f"Predictive entropy (mean ± std): {np.mean(unc_H):.4f} ± {np.std(unc_H):.4f}")
print(f"Variation ratio (mean ± std):    {np.mean(unc_V):.4f} ± {np.std(unc_V):.4f}")

if cov_points:
    print("\n=== Coverage–Risk (EM accuracy, after calibration) ===")
    for c, a in zip(cov_points, acc_points):
        print(f"Coverage {c:6.1%} | Accuracy {a:6.2%}")

print("\n==== BERTSCORE (unanswerable-aware, over MC samples) ====")
if np.isfinite(bs_mean):
    print(f"Mean of per-example BERTScore means: {bs_mean:.4f}")
    print(f"Mean of per-example BERTScore stds : {bs_std:.4f}")
    print(f"Mean of per-example BERTScore bests: {bs_best:.4f}")
    if cov_points_bs:
        print("\n=== Coverage–Risk (BERTScore mean as confidence, EM labels) ===")
        for c, a in zip(cov_points_bs, acc_points_bs):
            print(f"Coverage {c:6.1%} | Accuracy {a:6.2%}")
else:
    print("No items to score.")

print(f"\n=== Semantic Accuracy (BERTScore ≥ {BS_THRESHOLD:.2f}) ===")
if np.isfinite(sem_acc_overall):
    print(f"Overall semantic accuracy: {sem_acc_overall*100:.2f}%")
else:
    print("No semantic flags computed (insufficient data).")

if cov_points_sem:
    print("\n=== Coverage–Risk (Semantic accuracy; confidence = mean BERTScore) ===")
    for c, a in zip(cov_points_sem, acc_points_sem):
        print(f"Coverage {c:6.1%} | Accuracy {a*100:6.2f}%")

print(f"\n[Done] Total wall time: {time.perf_counter()-t_total0:.2f}s")
