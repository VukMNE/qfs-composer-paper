## This script generates summaries using the models defined in the paper.

# !pip install -U bitsandbytes
# !pip install transformers
# !pip uninstall -y datasets
# !pip install -U "datasets==2.20.0" "accelerate>=0.33.0" "huggingface-hub>=0.24.0"
# # (optional but helpful)
# !pip install -U "evaluate>=0.4.2"
# !pip -q install bert-score
# !pip install openai

# !pip install spacy
# !pip install spacy_udpipe


# from huggingface_hub import login
# login(token="*********")

# read source texts as datasets
import os
import json
from datasets import load_dataset, Features, Value, Sequence, DatasetDict
from qf_summ_composer import QfSummComposer
from tqdm import tqdm
import gc
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segmentrs:True"

# --- Experiment configuration ---
LLM_LIST = [
    #"gpt-4.1-mini"
    #"google/gemma-2-9b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    #"cjvt/GaMS-9B-Instruct"
]

QA_MODEL_ID = "VukDju/GaMS-9B-Instruct-QA-3ep"
QG_MODEL_ID = "VukDju/GaMS-9B-Instruct-QG-Full-3ep"
BERTSCORE_MODEL = "xlm-roberta-large"
CHUNK_SIZE = 350

DATA_FILES = {
    "test": "/content/qfs_slo_test_dataset_mini.json",
}

features = Features({
    "title": Value("string"),
    "subtitle": Value("string"),
    "whole_text": Value("string"),
    "source": Value("string"),
    "url": Value("string"),
    "author": Value("string"),
    "query_candidates": Sequence(Value("string")),  # <-- Add this line
})

test_dataset: DatasetDict = load_dataset(
    "json",
    data_files=DATA_FILES,
    features=features,
)
test_data = test_dataset["test"]

print(test_dataset)


SETTINGS = [
    #("decomposition", True,  "decomp_aug"),
    #("decomposition", False, "decomp_noaug"),
    #("named entity from query", False, "ner_noaug"),
    ("named entity from query", True,  "ner_aug"),
]

os.makedirs("summaries", exist_ok=True)


# for every LLM generate summaries for followng 4 cases:
    #   1. using decomposition for QG and augmented prompt,
    #   2. using decomposition for QG without augmented prompt,
    #   3. using named entities for QG and augmented prompt
    #   4. using named entities for QG without augmented prompt

for llm in LLM_LIST:
    llm_dir = os.path.join("summaries", llm.replace("/", "_"))
    os.makedirs(llm_dir, exist_ok=True)
    print(f"\n=== Running for LLM: {llm} ===")
    for qg_type, augment_prompt, setting_name in SETTINGS:
        print(f"  > Setting: {setting_name}")
        out_path = os.path.join(llm_dir, f"{setting_name}.json")
        results = []
        # Instantiate QfSummComposer for this setting
        qg_model = None if qg_type == "decomposition" else QG_MODEL_ID
        composer = QfSummComposer(
            decomposer_model_name="gpt-4.1-mini",
            bertscore_model_name=BERTSCORE_MODEL,
            qa_model_name=QA_MODEL_ID,
            qg_model_name=qg_model,
            summarizer_model_name=llm,
            question_generation_type=qg_type,
            chunk_size_in_chars=CHUNK_SIZE
        )
        for idx, ex in enumerate(tqdm(test_data, desc=f"{llm} | {setting_name}")):
            queries = ex.get("query_candidates", [])
            title = ex.get("title", "")
            subtitle = ex.get("subtitle", "")
            rest_of_text = ex.get("whole_text", "")
            source_text = title + "\n" + subtitle + "\n" + rest_of_text
            summary = composer.predict(queries[0], source_text, augment_prompt=augment_prompt)
            results.append({
                "id": idx,
                "query": queries[0],
                "summary": summary
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"    [{idx+1}/{len(test_data)}] Done")
        #  store summaries for every LLM and case as text files, so that I can just evaluate them later.

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        del composer  # Free memory before next setting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("All experiments completed. Summaries are saved in the 'summaries/' directory.")




# TODO evaluate summaries using QAGS, QuestEval and QA-Eval.