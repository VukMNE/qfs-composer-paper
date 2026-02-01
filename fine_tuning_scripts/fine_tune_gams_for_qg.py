import torch
import json
import gc
import os
from typing import Dict, List
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, AutoConfig
)
from datasets import load_from_disk, DatasetDict, Dataset

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("✅ PyTorch:", torch.__version__)

if torch.cuda.is_available():
    device = "cuda"
    print("✅ GPU is available:", torch.cuda.get_device_name(0))
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
    gc.collect()
else:
    device = "cpu"
    print("⚠️ No GPU detected! Training will be VERY slow.")

# Load dataset
dataset = load_from_disk("datasets/slovenian_squad_hf")

# Preprocessing for QG
MAX_NEW_Q_TOKENS = 48
MAX_TOTAL_LEN = 384
HALF_WINDOW_CHARS = 450

def center_window(context: str, ans_start: int, ans_text: str, half=HALF_WINDOW_CHARS) -> str:
    if ans_start is None or ans_start < 0:
        return context
    left = max(0, ans_start - half)
    right = min(len(context), ans_start + len(ans_text) + half)
    return context[left:right]

def build_example_qg(context: str, ans_text: str, ans_start: int, gold_question: str, tokenizer_qg) -> Dict[str, List[int]]:
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
    prompt_str = tokenizer_qg.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_text = f" {gold_question.strip()} [END]"
    prompt_ids = tokenizer_qg(prompt_str, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer_qg(target_text, add_special_tokens=False)["input_ids"]
    max_prompt_len = MAX_TOTAL_LEN - min(len(target_ids), MAX_NEW_Q_TOKENS + 16)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > MAX_TOTAL_LEN:
        cut = len(input_ids) - MAX_TOTAL_LEN
        input_ids = input_ids[cut:]
        attention_mask = attention_mask[cut:]
        labels = labels[cut:]
    pad = MAX_TOTAL_LEN - len(input_ids)
    if pad > 0:
        input_ids += [tokenizer_qg.pad_token_id] * pad
        attention_mask += [0] * pad
        labels += [-100] * pad
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def preprocess_qg(batch):
    contexts = batch["context"]
    questions = batch["question"]
    answers_list = batch["answers"]
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for c, q, ans in zip(contexts, questions, answers_list):
        ans_text = ans["text"] if isinstance(ans["text"], str) else ans["text"][0]
        ans_start = ans["answer_start"] if isinstance(ans["answer_start"], int) else ans["answer_start"][0]
        ex = build_example_qg(c, ans_text, ans_start, q, tokenizer_qg)
        out["input_ids"].append(ex["input_ids"])
        out["attention_mask"].append(ex["attention_mask"])
        out["labels"].append(ex["labels"])
    return out

# Load model & tokenizer
base_id = "cjvt/GaMS-2B-Instruct"
cfg = AutoConfig.from_pretrained(base_id)
cfg.hidden_dropout = 0.10
cfg.attention_dropout = 0.10

tokenizer_qg = AutoTokenizer.from_pretrained(base_id, use_fast=True)
if tokenizer_qg.pad_token is None:
    tokenizer_qg.pad_token = tokenizer_qg.eos_token

# Tokenize dataset
print("🔄 Tokenizing dataset for QG...")
qg_dataset = dataset.map(
    preprocess_qg,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing QG"
)
gc.collect()
torch.cuda.empty_cache()
print("✅ QG dataset tokenized successfully!")

# Load model with dropout enabled (for MC Dropout)
print(f"🔄 Loading QG model with eager attention...")
model_qg = AutoModelForCausalLM.from_pretrained(
    base_id,
    config=cfg,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="eager"
)
print(f"✅ QG model loaded.")

# Enable dropout for MC Dropout at inference
def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    print("✅ MC Dropout enabled (dropout layers set to train mode)")

# Enable gradient checkpointing for memory efficiency
model_qg.gradient_checkpointing_enable()

# Training arguments
training_args_qg = TrainingArguments(
    output_dir="models/slo-qg-GaMS-2B-Instruct-3ep",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    max_grad_norm=0.1,
    warmup_ratio=0.03,
    weight_decay=0.05,
    logging_steps=50,
    logging_dir="models/logs",
    fp16=False,
    bf16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=True,
    dataloader_num_workers=0,
    save_total_limit=1,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,
    ddp_find_unused_parameters=False,
    dataloader_drop_last=False,
    optim="adamw_torch_fused",
    adam_epsilon=1e-6,
    max_steps=-1,
    push_to_hub=False,
    eval_accumulation_steps=1,
    past_index=-1,
    run_name=None,
)

# Fix invalid generation config before training/saving
if hasattr(model_qg, "generation_config") and model_qg.generation_config is not None:
    gen_cfg = model_qg.generation_config
    # Remove temperature/top_p if do_sample is False
    if getattr(gen_cfg, "do_sample", False) is False:
        if hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = None
        if hasattr(gen_cfg, "top_p"):
            gen_cfg.top_p = None
        # Optionally, save the fixed config back to the model
        model_qg.generation_config = gen_cfg


print("🚀 Starting QG training...")

trainer_qg = Trainer(
    model=model_qg,
    args=training_args_qg,
    train_dataset=qg_dataset["train"],
    eval_dataset=qg_dataset["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

gc.collect()
torch.cuda.empty_cache()

trainer_qg.train()

# Save model
qg_model_path = "models/GaMS-2B-Instruct-QG-Full-3ep"
trainer_qg.save_model(qg_model_path)
tokenizer_qg.save_pretrained(qg_model_path)

print("✅ QG model saved successfully!")

# Example: Enable MC Dropout for inference
# enable_mc_dropout(model_qg)
# Now you can run multiple forward passes for MC Dropout uncertainty estimation

# Final memory cleanup
del model_qg
del trainer_qg
gc.collect()
torch.cuda.empty_cache()

print("✅ QG training completed and memory cleaned up!")