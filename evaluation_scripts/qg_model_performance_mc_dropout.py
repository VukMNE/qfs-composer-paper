# !pip -q uninstall -y gcsfs
# !pip -q install fsspec==2024.5.0 gcsfs==2024.5.0 s3fs==2024.5.0

# !pip -q install -U "transformers>=4.43.0" "accelerate>=0.33.0" "datasets==2.20.0" "huggingface-hub>=0.24.0" "evaluate>=0.4.2"
# !pip -q install -U rquge bitsandbytes

# !pip install git+https://github.com/VukMNE/RQUGE.git

from rquge_score import RQUGE

rquge_ = RQUGE(
    sp_scorer_path="VukDju/SloBERTA-SpanScorer-MOCHA",   # your FT folder or HF repo
    qa_model_path="VukDju/GaMS-2B-Instruct-QA-3epoch",            # full FT model, OR LoRA adapter
    device="cuda",
    language="sl",
)
rquge_

from transformers import StoppingCriteria, StoppingCriteriaList
import torch, re
from typing import List

def _encode_stop_sequences(tokenizer, stop_strs: List[str]) -> List[List[int]]:
    out = []
    for s in stop_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            out.append(ids)
    return out

class StopOnSubsequence(StoppingCriteria):
    """
    Batched stop: mark each row finished when it ends with any stop subsequence.
    Stop generation only when *all* rows are finished.
    Caches stop tensors per-device to avoid CPU/CUDA mismatch & repeated copies.
    """
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self._stops_cpu = [torch.tensor(seq, dtype=torch.long) for seq in stop_sequences]
        self._cache = {}      # device -> List[Tensor]
        self._finished = None # torch.BoolTensor per-batch, set on first call

    def _to_device(self, device: torch.device) -> List[torch.Tensor]:
        if device not in self._cache:
            self._cache[device] = [t.to(device) for t in self._stops_cpu]
        return self._cache[device]

    @staticmethod
    def _endswith(seq: torch.Tensor, suffix: torch.Tensor) -> bool:
        L = suffix.numel()
        if L == 0 or seq.size(0) < L:
            return False
        return torch.equal(seq[-L:], suffix)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        device = input_ids.device
        stops = self._to_device(device)

        B = input_ids.size(0)
        # Initialize/reset finished mask for this batch size
        if (self._finished is None) or (self._finished.numel() != B) or (self._finished.device != device):
            self._finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Update finished flags row-by-row
        for b in range(B):
            if not self._finished[b]:
                row = input_ids[b]
                for stop in stops:
                    if self._endswith(row, stop):
                        self._finished[b] = True
                        break

        # Stop only when *all* rows have hit a stop sequence
        return bool(self._finished.all())


# -*- coding: utf-8 -*-
# MC-Dropout for QG: QA-answerability variance & RQUGE variance


# ===== 1) Imports & setup =====
import os, gc, re, json, math, random, textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from datasets import load_dataset, Features, Value, Sequence, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

import evaluate  # used to load RQUGE
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

COLLECT_QG_SCORES = False   # <- new flag
COLLECT_QA_SCORES = True    # keep QA confidence variance if desired


# ===== 2) Paths / knobs (EDIT THESE) =====
# QG model checkpoint (your fine-tuned QG; local path or HF repo)
QG_CKPT = "VukDju/GaMS-2B-Instruct-QG-2epoch"  # change if needed
# A QA model to answer the sampled questions (your Slovene QA fits best)
QA_CKPT = "VukDju/GaMS-2B-Instruct-QA-3epoch"    # change to your best QA

# Dataset (SQuAD-style with fields: id, title, context, question, answers{text[], answer_start[]})
DATA_FILES = {
    "test": "/content/squad_special_qa_dataset.json"   # <-- set your eval file (can be same you used before)
}

# MC-Dropout sampling
K_MC = 10                     # MC passes per example
MC_BATCH = 5                   # do 8 masks in parallel (tune 4..16)
MAX_NEW_Q_TOKENS = 36         # QG decoding length cap (trained this way)
MAX_TOTAL_LEN = 384           # prompt+target limit (matches training)
HALF_WINDOW_CHARS = 450       # match your training windowing
DO_SAMPLE = False             # randomness only via dropout
TEMPERATURE = 1.0
TOP_P = 1.0

try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

# Output
DEBUG_JSONL = "/content/qg_mc_debug.jsonl"

# ===== 3) Load dataset (explicit schema is safest) =====
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

ds: DatasetDict = load_dataset("json", data_files=DATA_FILES, field="data", features=features)
eval_ds = ds["test"]
print("Eval examples:", len(eval_ds))

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # A100: bfloat16 is great
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# ===== 4) Tokenizers & Models (force eager attention + no cache for MC) =====
# ---- QG (generator) ----
qg_tok = AutoTokenizer.from_pretrained(QG_CKPT, use_fast=True)
if qg_tok.pad_token is None:
    qg_tok.pad_token = qg_tok.eos_token

# Ensure dropout is present in config (if you tweaked it during training, it’s already baked in)
qg_model = AutoModelForCausalLM.from_pretrained(
    QG_CKPT,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_cfg,
    attn_implementation="eager",  # ensure dropout in attention is active
)
qg_model.eval()

# ---- QA (answerer) ----
qa_tok = AutoTokenizer.from_pretrained(QA_CKPT, use_fast=True)
if qa_tok.pad_token is None:
    qa_tok.pad_token = qa_tok.eos_token

qa_model = AutoModelForCausalLM.from_pretrained(
    QA_CKPT,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
qa_model.eval()

# ===== 5) Utilities (prompting & text cleaning) =====
def center_window(context: str, ans_start: int, ans_text: str, half=HALF_WINDOW_CHARS) -> str:
    if ans_start is None or ans_start < 0:
        return context
    left = max(0, ans_start - half)
    right = min(len(context), ans_start + len(ans_text) + half)
    return context[left:right]

# ----- QG prompt: mirror your training template (context + <ans> tags) -----
def build_qg_prompt(context: str, ans_text: str, ans_start: int, tokenizer) -> str:
    windowed = center_window(context, ans_start, ans_text)
    if ans_text in windowed:
        windowed = windowed.replace(ans_text, f"<ans>{ans_text}</ans>", 1)
    user_content = (
        "Na podlagi naslednjega besedila in podanega odgovora generiraj samo eno vprašanje,"
        "na katerega je ta podani odgovor pravilen in smiseln izključno v kontekstu tega besedila."
        "Vprašanje naj bo oblikovano tako, da je prav podani odgovor (in ne katerikoli drug) edini pravilen odgovor."
        "Zaključi vprašanje z oznako [END]. "
        f"\nBesedilo: {windowed}\nOdgovor: {ans_text}\nVprašanje:"
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|asistent|assistantu|asistentu|model|bot)\s*[:\-]?\s*", re.IGNORECASE)
END_SPLIT_RE   = re.compile(r"\[?\s*END\s*\]?", re.IGNORECASE)
ANS_PREFIX_RE  = re.compile(r"^\s*Odgovor\s*:\s*", re.IGNORECASE)

def _strip_after_terminator(s: str) -> str:
    m = END_SPLIT_RE.search(s)
    return s[:m.start()] if m else s

def clean_question_text(s: str) -> str:
    s = s or ""
    s = ROLE_PREFIX_RE.sub("", s, count=1)
    s = _strip_after_terminator(s)
    s = s.replace("\u200b"," ").replace("\r"," ").replace("\n"," ")
    return " ".join(s.split())

def clean_answer_text(s: str) -> str:
    s = s or ""
    s = ANS_PREFIX_RE.sub("", s, count=1)
    s = ROLE_PREFIX_RE.sub("", s, count=1)
    s = _strip_after_terminator(s)
    s = s.replace("\u200b"," ").replace("\r"," ").replace("\n"," ")
    return " ".join(s.split())


NO_ANS_TOKEN = "<no_answer>"

def build_qa_prompt_plain(question: str, context: str, tokenizer) -> str:
    return (
        "Na podlagi podanega vprašanja in besedila generiraj samo en smiseln in pravilen odgovor, "
        "ki izhaja izključno iz konteksta tega besedila. "
        "Odgovor naj bo jasen, natančen in kratek (le nekaj besed). "
        f"Če na vprašanje ni mogoče odgovoriti na podlagi predloženega besedila, vrni {NO_ANS_TOKEN}."
        f"\nVprašanje: {question}\nBesedilo: {context}\nOdgovor:"
    )

def is_no_answer(ans: str) -> bool:
    return NO_ANS_TOKEN in (ans or "").lower()

def mc_qg_samples_batched(context: str, ans_text: str, ans_start: int, k: int) -> Dict[str, Any]:
    prompt = build_qg_prompt(context, ans_text, ans_start, qg_tok)
    questions, mean_lps = [], []

    # prepare stop sequences for "[END]"
    qg_stops = _encode_stop_sequences(qg_tok, ["[END]"])
    stop_criteria = StoppingCriteriaList([StopOnSubsequence(qg_stops)]) if qg_stops else None

    qg_model.train()  # activate dropout
    with torch.no_grad():
        done = 0
        while done < k:
            b = min(MC_BATCH, k - done)
            toks = qg_tok(
                [prompt] * b,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=MAX_TOTAL_LEN,
            ).to(qg_model.device)

            out = qg_model.generate(
                **toks,
                max_new_tokens=MAX_NEW_Q_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,  # keep for speed
                return_dict_in_generate=True,
                output_scores=COLLECT_QG_SCORES,
                pad_token_id=qg_tok.pad_token_id or qg_tok.eos_token_id,
                stopping_criteria=stop_criteria,
            )

            prompt_len = toks["input_ids"].shape[1]
            gen_ids = out.sequences[:, prompt_len:]   # (B, T*)
            B = gen_ids.size(0)                       # <-- don't touch out.scores when it's None

            # Decode each row
            for row in gen_ids:
                q_txt = qg_tok.decode(row, skip_special_tokens=True)
                questions.append(clean_question_text(q_txt))

            # If we’re not collecting QG scores, just fill NaNs (or drop the field)
            if COLLECT_QG_SCORES and getattr(out, "scores", None) is not None:
                # (optional) compute mean logprob here like before
                lpsum = torch.zeros(B, dtype=torch.float32, device=gen_ids.device)
                count = torch.zeros(B, dtype=torch.int32, device=gen_ids.device)
                pad_id = qg_tok.pad_token_id or qg_tok.eos_token_id
                T = len(out.scores)
                for t in range(T):
                    step_logp = out.scores[t].log_softmax(dim=-1)  # (B, V)
                    tids = gen_ids[:, t]
                    lpsum += step_logp[torch.arange(B, device=gen_ids.device), tids]
                    count += (tids != pad_id).to(torch.int32)
                mean_lp = (lpsum / torch.clamp(count, min=1)).detach().cpu().numpy().astype(float).tolist()
                mean_lps.extend(mean_lp)
            else:
                mean_lps.extend([float("nan")] * B)

            done += b

    qg_model.eval()
    print('Questions:')
    print(questions)
    return {"questions": questions, "qg_mean_logprob": mean_lps}


def generate_once(model, tok, prompt: str, max_new: int) -> Tuple[str, float]:
    # Tokenize the prompt
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_cache=False,                   # critical for MC-Dropout
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )

    # === ONLY decode the generated tail, not the whole sequence ===
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    # Mean token logprob over generated tokens
    lps = []
    for step_logits, tid in zip(out.scores, gen_ids):
        lp = step_logits.log_softmax(dim=-1)[0, tid.item()].item()
        lps.append(lp)
    mean_lp = float(np.mean(lps)) if lps else -1e9

    return text, mean_lp

def qa_answer_on(context: str, question: str) -> Tuple[str, float]:
    prompt = build_qa_prompt_plain(question, context, qa_tok)

    qa_stops = _encode_stop_sequences(qa_tok, ["[END]"])   # model sometimes emits it
    stop_criteria = StoppingCriteriaList([StopOnSubsequence(qa_stops)]) if qa_stops else None

    qa_model.train()
    with torch.no_grad():
        inputs = qa_tok(prompt, return_tensors="pt", add_special_tokens=False).to(qa_model.device)
        out = qa_model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=qa_tok.pad_token_id or qa_tok.eos_token_id,
            stopping_criteria=stop_criteria,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0][prompt_len:]
    ans = qa_tok.decode(gen_ids, skip_special_tokens=True)
    ans = clean_answer_text(ans)

    lps = []
    for step_logits, tid in zip(out.scores, gen_ids):
        lps.append(step_logits.log_softmax(dim=-1)[0, tid.item()].item())
    return ans, (float(np.mean(lps)) if lps else -1e9)




# ===== 9) Main loop =====
if os.path.exists(DEBUG_JSONL):
    open(DEBUG_JSONL, "w").close()

per_example_rows = []  # keep for aggregate stats

pbar = tqdm(eval_ds, desc="QG MC-Dropout evaluation")
for i, ex in enumerate(pbar):
    print('-------------------------------')
    ctx = ex["context"]
    gold_ans_text = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
    gold_ans_start = int(ex["answers"]["answer_start"][0]) if ex["answers"]["answer_start"] else -1

    # (A) MC-sample questions
    mc = mc_qg_samples_batched(ctx, gold_ans_text, gold_ans_start, K_MC)
    questions = mc["questions"]

    # (B) For each sampled question: run QA once; record confidence and answerability
    qa_confs = []
    ans_flags = []   # 1 if answerable (non-<no_answer>), else 0
    qa_preds = []

    for q in questions:
        pred_ans, lp = qa_answer_on(ctx, q)
        qa_confs.append(lp)
        ans_flags.append(0 if is_no_answer(pred_ans) else 1)
        qa_preds.append(pred_ans)
        print('QA model answered the question...')

    # (C) RQUGE per-sample (reference-free, needs context + answer span + question)

    rquge_scores = []
    for question in questions:
      try:
        rq_score, pred_ans = rquge_.scorer(context=ctx, pred_question=question, gold_answer=gold_ans_text, max_new_tokens=30)
        rquge_scores.append(rq_score)
      except Exception as e:
        # If something goes wrong for one item, record NaN so we can keep going
        score_f = float("nan")
        pred_ans = ""
        rquge_scores.append(score_f)






    # Stats per example
    qa_conf_var = float(np.var(qa_confs, ddof=1)) if len(qa_confs) > 1 else 0.0
    ans_rate = float(np.mean(ans_flags)) if ans_flags else 0.0
    ans_rate_var = float(np.var(ans_flags, ddof=1)) if len(ans_flags) > 1 else 0.0

    rquge_arr = np.array(rquge_scores, dtype=np.float64)
    rquge_mean = float(rquge_arr.mean()) if len(rquge_arr) else float("nan")
    rquge_var  = float(rquge_arr.var(ddof=1)) if len(rquge_arr) > 1 else 0.0
    rquge_std  = float(rquge_arr.std(ddof=1)) if len(rquge_arr) > 1 else 0.0

    row = {
        "index": i,
        "context": ctx,
        "gold_answer": gold_ans_text,
        "gold_answer_start": gold_ans_start,
        "mc_questions": questions,
        "mc_qg_mean_logprob": mc["qg_mean_logprob"],
        "qa_preds": qa_preds,
        "qa_confidences": qa_confs,
        "answerable_flags": ans_flags,
        "qa_conf_variance": qa_conf_var,
        "answerability_rate": ans_rate,
        "answerability_variance": ans_rate_var,
        "rquge_scores": rquge_scores,
        "rquge_mean": rquge_mean,
        "rquge_std": rquge_std,
        "rquge_variance": rquge_var,
    }
    per_example_rows.append(row)

    # stream to JSONL
    with open(DEBUG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ===== 10) Aggregate report =====
qa_conf_vars   = [r["qa_conf_variance"]      for r in per_example_rows]
ans_rate_vars  = [r["answerability_variance"] for r in per_example_rows]
ans_rates      = [r["answerability_rate"]     for r in per_example_rows]
rquge_means    = [r["rquge_mean"]             for r in per_example_rows if not math.isnan(r["rquge_mean"])]
rquge_vars     = [r["rquge_variance"]         for r in per_example_rows]

def safe_mean(x): return float(np.mean(x)) if len(x) else float("nan")
def safe_std(x):  return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

print("\n==================== SUMMARY ====================")
print(f"Examples evaluated: {len(per_example_rows)}")
print("\n--- QA-answerability (across MC samples) ---")
print(f"Mean answerability rate:         {safe_mean(ans_rates):.4f}")
print(f"Mean variance of QA confidence:  {safe_mean(qa_conf_vars):.6f}  (std={safe_std(qa_conf_vars):.6f})")
print(f"Mean variance of answerability:  {safe_mean(ans_rate_vars):.6f} (std={safe_std(ans_rate_vars):.6f})")
print("\n--- RQUGE (across MC samples) ---")
print(f"Mean of per-example RQUGE means: {safe_mean(rquge_means):.4f}")
print(f"Mean of per-example RQUGE vars : {safe_mean(rquge_vars):.6f}  (std={safe_std(rquge_vars):.6f})")
print("\nDebug JSONL saved to:", DEBUG_JSONL)





# Additional information:
import json, math
import numpy as np

# === Set your path here ===
PATH = "/content/qg_mc_debug.jsonl"  # change if needed

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

rows = load_jsonl(PATH)

per_ex_means = []
per_ex_stds  = []

for ex in rows:
    gold = (ex.get("gold_answer") or "").strip()
    if gold == "":
        continue  # skip unanswerable/empty-gold examples

    scores = ex.get("rquge_scores", [])
    # keep only finite numbers
    scores = [float(s) for s in scores if isinstance(s, (int, float)) and math.isfinite(float(s))]
    if not scores:
        continue

    # per-example stats
    m = float(np.mean(scores))
    if len(scores) >= 2:
        s = float(np.std(scores, ddof=1))
    else:
        s = 0.0

    per_ex_means.append(m)
    per_ex_stds.append(s)

# Aggregate metrics
mean_of_means = float(np.mean(per_ex_means)) if per_ex_means else float("nan")
mean_of_stds  = float(np.mean(per_ex_stds))  if per_ex_stds  else float("nan")

print(f"Examples used: {len(per_ex_means)}")
print(f"Mean of per-example RQUGE means: {mean_of_means:.4f}")
print(f"Mean of per-example RQUGE standard deviations: {mean_of_stds:.4f}")