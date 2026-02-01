# !pip install -U bitsandbytes
# !pip install transformers
# !pip uninstall -y datasets
# !pip install -U "datasets==2.20.0" "accelerate>=0.33.0" "huggingface-hub>=0.24.0"
# # (optional but helpful)
# !pip install -U "evaluate>=0.4.2"
# !pip -q install bert-score
# !pip install openai
# !pip install editdistance
# !pip install spacy
# !pip install spacy_udpipe
# !pip install git+https://github.com/VukMNE/RQUGE.git

# TODO here we will evaluate the summaries using qags, questeval and other metrics.

# summaries are stored in summaries/ folder.

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset, Features, Value, Sequence, DatasetDict
from tqdm import tqdm
from qags.qags import qags_eval



LLM_LIST = [
    "gpt-4.1-mini",
    "google/gemma-2-9b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "cjvt/GaMS-9B-Instruct"
]

SETTINGS = [
    ("decomposition", True,  "decomp_aug"),
    ("decomposition", False, "decomp_noaug"),
    ("named entity from query", True,  "ner_aug"),
    ("named entity from query", False, "ner_noaug"),
]

QA_MODEL_ID = "VukDju/GaMS-9B-Instruct-QA-3ep"
QG_MODEL_ID = "VukDju/GaMS-9B-Instruct-QG-Full-3ep"



DATA_FILES = {
    # If you only have one file now, you can point both to the same path.
    #"test": "/content/qfs_slo_single_example.json",
    "test": "/content/qfs_slo_test_dataset.json",
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



sum_dt_features = Features({
    "id": Value("int32"),
    "query": Value("string"),
    "summary": Value("string")
})




def evaluate_summaries(model_name):
    SUMM_FILES = {
        #"decomp_aug": "/content/drive/MyDrive/summaries/" + model_name + "/decomp_aug.json",
        #"decomp_noaug": "/content/drive/MyDrive/summaries/" + model_name + "/decomp_noaug.json",
        "ner_aug": "/content/drive/MyDrive/summaries/" + model_name + "/ner_aug.json",
        "ner_noaug": "/content/drive/MyDrive/summaries/" + model_name + "/ner_noaug.json",
    }


    sum_dataset: DatasetDict = load_dataset(
        "json",
        data_files=SUMM_FILES,
        features=sum_dt_features,
    )

    qags_scores = {
        "decomp_aug": [],
        "decomp_noaug": [],
        "ner_aug": [],
        "ner_noaug": []
    }

    source_texts = []


    for idx, ex in enumerate(tqdm(test_data, desc=f"Evaluating {model_name}")):
        # You may want to use a specific field as query, e.g. ex["title"] or ex["subtitle"]
        queries = ex.get("query_candidates", [])  # or use a custom query field
        title = ex.get("title", "")
        subtitle = ex.get("subtitle", "")
        rest_of_text = ex.get("whole_text", "")
        source_text = title + "\n" + subtitle + "\n" + rest_of_text
        source_texts.append(source_text)



    for setting_name in SUMM_FILES.keys():
        summaries = [sum_dataset[setting_name][idx]["summary"] for idx in range(len(test_data))]
        # Evaluate using QAGS

        f1_qags, em_qags, edit_distance_qags = qags_eval(
            summaries=summaries,
            source_texts=source_texts,
            out_file=f"qags_scores_{model_name}_{setting_name}.json"
        )
        # Evaluate using QuestEval
        #questeval_score = questeval.compute(predictions=[summ], references=[source_text], batch_size=1)

        # Store or print the results
        print(f"Model: {model_name}, Setting: {setting_name}, Example ID: {idx}")
        print(f"  QAGS Scores:  F1 : {f1_qags:.4f}, EM: {em_qags:.4f}, Edit Distance: {edit_distance_qags:.4f}")
        # print(f"  QuestEval Score: {questeval_score['score']:.4f}")

        print()


for llm in LLM_LIST:
    llm_dir = os.path.join("summaries", llm.replace("/", "_"))
    print(f"\n=== Running for LLM: {llm} ===")
    evaluate_summaries(llm.replace("/", "_"))
