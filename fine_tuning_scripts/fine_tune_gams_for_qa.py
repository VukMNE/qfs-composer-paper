import torch
import json
from random import randint
from typing import Dict, List
import gc
import os
import sys

# Memory optimization imports
from transformers import (
    GemmaConfig, GemmaForCausalLM, GemmaTokenizer, 
    Trainer, TrainingArguments, EarlyStoppingCallback, 
    AutoModelForCausalLM, AutoTokenizer, TrainerCallback, AutoConfig
)
from datasets import DatasetDict, Dataset, load_from_disk

# Enable memory efficient attention and other optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("✅ PyTorch:", torch.__version__)

if torch.cuda.is_available():
    device = "cuda"
    print("✅ GPU is available:", torch.cuda.get_device_name(0))
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
else:
    device = "cpu"
    print("⚠️ No GPU detected! Training will be VERY slow.")

# Set path to dataset
squad_file_path = "datasets/squad2-slo-mt-train.json"
squad_val_file_path = "datasets/squad_slo_mini_test.json"

# Load dataset
with open(squad_file_path, "r", encoding="utf-8") as f:
    squad_data = json.load(f)

with open(squad_val_file_path, "r", encoding="utf-8") as fv:
    squad_validation_data = json.load(fv)

print("✅ Dataset loaded successfully!")

# Preprocess SQuAD dataset (same as your original)
# def preprocess_squad(data):
#     contexts, questions, answers = [], [], []
#     for article in data["data"]:
#         question = article["question"]
#         context = article["context"]
#         if len(article["answers"]["text"]) > 0:
#             contexts.append(context)
#             questions.append(question)
#             answers.append({
#                 "text": article["answers"]["text"][0], 
#                 "answer_start": article["answers"]["answer_start"][0]
#             })
#     return {"context": contexts, "question": questions, "answers": answers}

def preprocess_squad(data):
    contexts, questions, answers = [], [], []
    for article in data["data"]:
        question = article["question"]
        context = article["context"]

        # SQuAD-v2 style: may have zero answers
        txt_list = article["answers"]["text"]
        start_list = article["answers"]["answer_start"]

        if isinstance(txt_list, list) and len(txt_list) > 0:
            ans_text = str(txt_list[0]).strip()
            ans_start = int(start_list[0]) if isinstance(start_list, list) and len(start_list) > 0 else -1
        else:
            ans_text = ""          # <-- keep unanswerable
            ans_start = -1

        contexts.append(context)
        questions.append(question)
        answers.append({"text": ans_text, "answer_start": ans_start})
    return {"context": contexts, "question": questions, "answers": answers}


# Convert and save dataset
squad_dataset = preprocess_squad(squad_data)
squad_validation_dataset = preprocess_squad(squad_validation_data)
hf_dataset = DatasetDict({
    "train": Dataset.from_dict(squad_dataset), 
    "test": Dataset.from_dict(squad_validation_dataset)
})

hf_dataset.save_to_disk("datasets/slovenian_squad_hf")
print("✅ Dataset converted and saved!")

# Memory-efficient model loading
base_id = "cjvt/GaMS-9B-Instruct"

# 1) Load and tweak config so the base model actually has dropout
cfg = AutoConfig.from_pretrained(base_id)
# Gemma/GaMS defaults are often 0.0; set small but non-zero rates
cfg.hidden_dropout = 0.10          # MLP / residual dropout
cfg.attention_dropout = 0.10       # attention dropout (will auto-disable at eval)

model_qa = None
attention_implementations = ["eager"]


for attn_impl in attention_implementations:
    try:
        print(f"🔄 Trying to load model with {attn_impl} attention...")
        model_qa = AutoModelForCausalLM.from_pretrained(
            base_id,
            config=cfg,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        print(f"✅ Successfully loaded model with {attn_impl} attention")
        break
    except Exception as e:
        print(f"⚠️  Failed to load with {attn_impl}: {str(e)}")
        if model_qa is not None:
            del model_qa
            torch.cuda.empty_cache()
        continue


if model_qa is None:
    # Final fallback - load without specifying attention implementation
    print("🔄 Final fallback: loading without attention specification...")
    model_qa = AutoModelForCausalLM.from_pretrained(
        base_id,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


# Enable gradient checkpointing for memory efficiency
model_qa.gradient_checkpointing_enable()

def summarize_dropout(m):
    n, nonzero = 0, 0
    ps = []
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            n += 1
            if mod.p > 0:
                nonzero += 1
                ps.append(mod.p)
    print(f"🔎 nn.Dropout modules: {n}, non-zero p: {nonzero}, distinct p values: {sorted(set(ps))}")

summarize_dropout(model_qa)
# Expect nonzero > 0 and p≈0.10 listed


# Load tokenizer
tokenizer_qa = AutoTokenizer.from_pretrained(base_id, use_fast=True)
if tokenizer_qa.pad_token is None:
    tokenizer_qa.pad_token = tokenizer_qa.eos_token

print("✅ Model and tokenizer loaded successfully!")

# Load dataset
dataset = load_from_disk("datasets/slovenian_squad_hf")

# Preprocessing functions (same as your original but with memory optimizations)
MAX_NEW_ANS_TOKENS = 24
MAX_TOTAL_LEN = 384  # Reduced from 384 to save memory
HALF_WINDOW_CHARS = 350  # Reduced from 450


def center_window_qa(context: str, ans_start: int, ans_text: str, half=HALF_WINDOW_CHARS) -> str:
    if ans_start is None or ans_start < 0:
        return context
    left = max(0, ans_start - half)
    right = min(len(context), ans_start + len(ans_text) + half)
    return context[left:right]

def _extract_answer(ans_obj):
    if ans_obj is None:
        return "", -1

    txt = ans_obj.get("text", "")
    start = ans_obj.get("answer_start", -1)

    if isinstance(txt, list):
        if len(txt) > 0:
            ans_text = str(txt[0])
            if isinstance(start, list):
                ans_start = int(start[0]) if start else -1
            else:
                ans_start = int(start) if start is not None else -1
            return ans_text, ans_start
        else:
            return "", -1

    if isinstance(txt, str):
        ans_text = txt
        try:
            ans_start = int(start) if start is not None else -1
        except Exception:
            ans_start = -1
        return ans_text, ans_start

    return "", -1

# ------ once, after loading tokenizer/model ------
NO_ANS_TOKEN = "<no_answer>"
added = tokenizer_qa.add_special_tokens({"additional_special_tokens": ["[END]", NO_ANS_TOKEN]})
if added > 0:
    model_qa.resize_token_embeddings(len(tokenizer_qa))
END_ID = tokenizer_qa.convert_tokens_to_ids("[END]")

MAX_TOTAL_LEN = 384
MAX_NEW_ANS_TOKENS = 24

def build_example_qa(context: str, question: str, ans_text: str):
    # 1) target text
    ans = ans_text.strip() if (ans_text and ans_text.strip()) else NO_ANS_TOKEN
    target_text = f" {ans}\n[END]"
    target_ids = tokenizer_qa(target_text, add_special_tokens=False)["input_ids"]

    # 2) helper to render chat-prompt *with* template tokens
    def render_ctx(ctx_str: str):
        user = (
            "Na podlagi podanega vprašanja in besedila generiraj samo en smiseln in pravilen odgovor, "
            "ki izhaja izključno iz konteksta tega besedila. "
            "Odgovor naj bo jasen, natančen in kratek (le nekaj besed). "
            f"Če na vprašanje ni mogoče odgovoriti na podlagi predloženega besedila, vrni {NO_ANS_TOKEN}."
            f"\n\nVprašanje: {question}\nBesedilo: {ctx_str}\nOdgovor:"
        )
        messages = [{"role":"user","content": user}]
        return tokenizer_qa.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )["input_ids"]

    # 3) find the longest context that fits (binary search over chars)
    lo, hi = 0, len(context)
    best_prompt_ids = None
    budget = MAX_TOTAL_LEN - len(target_ids)
    while lo <= hi:
        mid = (lo + hi) // 2
        prompt_ids = render_ctx(context[:mid])
        if len(prompt_ids) <= budget:
            best_prompt_ids = prompt_ids
            lo = mid + 1
        else:
            hi = mid - 1

    if best_prompt_ids is None:
        best_prompt_ids = render_ctx("")  # fall back to no context

    input_ids = best_prompt_ids + target_ids
    labels = [-100] * len(best_prompt_ids) + target_ids
    attn_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


# def build_example_qa(context: str, question: str, ans_text: str, ans_start: int) -> Dict[str, List[int]]:
#     # Crop context around answer
#     windowed = center_window_qa(context, ans_start, ans_text)

#     # Build chat-style user turn
#     user = (
#         "Na podlagi podanega vprašanja in besedila generiraj samo en smiseln in pravilen odgovor, "
#         "ki izhaja izključno iz konteksta tega besedila. "
#         "Odgovor naj bo jasen, natančen in kratek (le nekaj besed). "
#         f"Če na vprašanje ni mogoče odgovoriti na podlagi predloženega besedila, vrni {NO_ANS_TOKEN}."
#         f"\n\nVprašanje: {question}\nBesedilo: {windowed}\nOdgovor:"
#     )
#     messages = [{"role": "user", "content": user}]
#     prompt_str = tokenizer_qa.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     # NEW: explicit no-answer token (instead of empty)
#     if ans_text.strip():
#         target_text = f" {ans_text.strip()}\n[END]"
#     else:
#         target_text = f" {NO_ANS_TOKEN}\n[END]"

#     # Tokenize
#     prompt_ids = tokenizer_qa(prompt_str, add_special_tokens=False)["input_ids"]
#     target_ids = tokenizer_qa(target_text, add_special_tokens=False)["input_ids"]

#     # Keep budget for target; truncate prompt from LEFT if needed
#     max_prompt_len = MAX_TOTAL_LEN - min(len(target_ids), MAX_NEW_ANS_TOKENS + 16)
#     if len(prompt_ids) > max_prompt_len:
#         prompt_ids = prompt_ids[-max_prompt_len:]

#     input_ids = prompt_ids + target_ids
#     attention_mask = [1] * len(input_ids)
#     labels = [-100] * len(prompt_ids) + target_ids

#     # Final clip if still too long
#     if len(input_ids) > MAX_TOTAL_LEN:
#         cut = len(input_ids) - MAX_TOTAL_LEN
#         input_ids = input_ids[cut:]
#         attention_mask = attention_mask[cut:]
#         labels = labels[cut:]

#     # Pad
#     pad = MAX_TOTAL_LEN - len(input_ids)
#     if pad > 0:
#         input_ids += [tokenizer_qa.pad_token_id] * pad
#         attention_mask += [0] * pad
#         labels += [-100] * pad

#     return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def preprocess_qa(batch):
    contexts = batch["context"]
    questions = batch["question"]
    answers_li = batch["answers"]

    out = {"input_ids": [], "attention_mask": [], "labels": []}

    for c, q, ans in zip(contexts, questions, answers_li):
        ans_text, ans_start = _extract_answer(ans)
        ex = build_example_qa(c, q, ans_text)
        out["input_ids"].append(ex["input_ids"])
        out["attention_mask"].append(ex["attention_mask"])
        out["labels"].append(ex["labels"])

    return out

# Tokenize dataset
print("🔄 Tokenizing dataset...")
qa_dataset = dataset.map(
    preprocess_qa, 
    batched=True, 
    remove_columns=dataset["train"].column_names,  # Remove original columns to save memory
    desc="Tokenizing"
)

# Clear memory after preprocessing
gc.collect()
torch.cuda.empty_cache()

print("✅ Dataset tokenized successfully!")

# Custom trainer class for memory optimization
class MemoryEfficientTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Clear cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        # Clear cache after backward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss.detach() / self.args.gradient_accumulation_steps

# Memory-efficient training arguments
training_args_qa = TrainingArguments(
    output_dir="models/slo-qa-GaMS-9B-Instruct-full-3ep",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-6,  # Lower learning rate for full model fine-tuning
    per_device_train_batch_size=1,  # Keep batch size at 1
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    gradient_accumulation_steps=16,  # Increased to simulate larger batch
    max_grad_norm=0.1,  # Lower gradient clipping for stability
    warmup_ratio=0.1,  # More warmup for stability
    weight_decay=0.01,  # Small weight decay for regularization
    logging_steps=25,  # Changed from 25 to 50 to match your request
    logging_dir="models/logs",
    fp16=False,  # Use fp16 for memory efficiency
    bf16=True,
    dataloader_pin_memory=False,  # Reduce memory usage
    remove_unused_columns=True,
    dataloader_num_workers=0,  # Reduce CPU memory usage
    save_total_limit=1,  # Keep only 1 checkpoint to save disk space
    load_best_model_at_end=False,  # Disable to save memory during training
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,  # Disable wandb/tensorboard to save memory
    # Memory optimization flags
    ddp_find_unused_parameters=False,
    dataloader_drop_last=True,
    optim="adamw_bnb_8bit",
    optim_args="adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8",
    adam_epsilon=1e-6,
    max_steps=-1,
    push_to_hub=False,  # Disable during training to save memory, push manually after
    # Additional memory optimizations
    eval_accumulation_steps=1,  # Process eval in smaller chunks
    past_index=-1,
    run_name=None
)

# Fix invalid generation config before training/saving
if hasattr(model_qa, "generation_config") and model_qa.generation_config is not None:
    gen_cfg = model_qa.generation_config
    # Remove temperature/top_p if do_sample is False
    if getattr(gen_cfg, "do_sample", False) is False:
        if hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = None
        if hasattr(gen_cfg, "top_p"):
            gen_cfg.top_p = None
        # Optionally, save the fixed config back to the model
        model_qa.generation_config = gen_cfg

print("🚀 Starting training...")

# Use standard trainer instead of custom one to avoid compatibility issues
trainer_qa = Trainer(
    model=model_qa,
    args=training_args_qa,
    train_dataset=qa_dataset["train"],
    eval_dataset=qa_dataset["test"]
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# Clear memory before training
gc.collect()
torch.cuda.empty_cache()

trainer_qa.train()

# Save model
qg_model_path = "models/GaMS-9B-Instruct-QA-Full-3ep"
trainer_qa.save_model(qg_model_path)
tokenizer_qa.save_pretrained(qg_model_path)

print("✅ Full model saved successfully!")

# Final memory cleanup
del model_qa
del trainer_qa
gc.collect()
torch.cuda.empty_cache()

print("✅ Training completed and memory cleaned up!")