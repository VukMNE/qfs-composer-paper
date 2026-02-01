from typing import List
from decomposer.decomposer import QueryDecomposer
from chunker.large_document_chunk_scorer import LargeDocumentChunkScorer
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import re
from ner.ne_extractor import NE_Extractor


QA_MODEL_ID = "VukDju/GaMS-9B-Instruct-QA-3ep"  # <-- your saved model dir from training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123
QA_MAX_NEW_TOKENS = 24 # increase to 48 for longer answers
DO_SAMPLE = False          # keep greedy; dropout is the only randomness
TEMPERATURE = 1.0
TOP_P = 1.0

ROLE_PREFIX_RE = re.compile(r"^\s*(model|assistant|asistent|assistantu|asistentu)?\s*[:\-]?\s*", re.IGNORECASE)
END_TOK_RE     = re.compile(r"\[?\s*END\s*\]?", re.IGNORECASE)

class QfSummComposer():

    def __init__(self,
                 decomposer_model_name: str,
                 bertscore_model_name: str,
                 qa_model_name: str,
                 qg_model_name: str,
                 summarizer_model_name: str,
                 question_generation_type: str, # can be either 'decomposition' or 'named entity from query'
                 chunk_size_in_chars: int):
        

        self.decomposer = QueryDecomposer(model=decomposer_model_name)
        self.chunk_scorer = LargeDocumentChunkScorer(chunk_size_in_chars, bertscore_model=bertscore_model_name)
        
        if question_generation_type not in ['decomposition', 'named entity from query']:
            raise ValueError("question_generation_type must be either 'decomposition' or 'named entity from query'")
        
        self.question_generation_type = question_generation_type

        if question_generation_type == 'named entity from query':
            self.ne_extractor = NE_Extractor(language='sl') 

        
                # -----------------------------
        # Tokenizer & Model (force eager attention so dropout is honored)
        # -----------------------------

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # A100: bfloat16 is great
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name, use_fast=True)
        if self.qa_tokenizer.pad_token is None:
            self.qa_tokenizer.pad_token = self.qa_tokenizer.eos_token


        # We still force 'eager' attention to ensure attention-dropout isn't bypassed.
        self.qa_model = AutoModelForCausalLM.from_pretrained(
            qa_model_name,
            device_map="auto",
            quantization_config=bnb_cfg,
            attn_implementation="eager",
        )

        self.qa_model.eval()
        self.qa_model_name = qa_model_name

        # initiliaze question generation model here if needed
        if qg_model_name != None and question_generation_type == 'named entity from query':
            
            self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name, use_fast=True)
            self.qg_model = AutoModelForCausalLM.from_pretrained(
                qg_model_name,
                device_map="auto",
                quantization_config=bnb_cfg,
                attn_implementation="eager"
            )
            self.qg_model.eval()
            self.qg_model_name = qg_model_name


        self.summarizer_model_name = summarizer_model_name
        # Detect summary model provider
        if any(summarizer_model_name.startswith(x) for x in ["gpt-", "o3", "o4"]):
            self.summary_provider = "openai"
            from openai import OpenAI
            self.openai_client = OpenAI(api_key="sk-**********")
        elif summarizer_model_name.startswith("google/gemma"):
            # works only for gemma instruction models that end with "it" like "google/gemma-2-9b-it"
            self.summary_provider = "gemma"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            self.gemma_pipe = pipeline(
                "text-generation",
                model=summarizer_model_name,
                model_kwargs={"dtype": dtype}, # used to be "torch_dtype"
                device_map="auto"
            )
        elif summarizer_model_name.startswith("meta-llama"):
            # works only for llama instruction models 
            self.summary_provider = "llama"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.llama_pipe = pipeline(
                "text-generation",
                model=summarizer_model_name,
                model_kwargs={"dtype": dtype},
                device_map="auto"
            )
        else:
            self.summary_provider = "hf"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.hf_pipe = pipeline(
                "text-generation",
                model=summarizer_model_name,
                model_kwargs={"dtype": dtype},
                device_map="auto"
            )

    def generate_summary(self, prompt: str, max_new_tokens: int = 512) -> str:
        if self.summary_provider == "openai":
            r = self.openai_client.responses.create(model=self.summarizer_model_name, input=prompt)
            return r.output_text.strip()
        elif self.summary_provider == "gemma":
            messages = [
                {"role": "user", "content": prompt},
            ]
            outs = self.gemma_pipe(messages, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, num_return_sequences=1)
            return outs[0]["generated_text"][-1]["content"].strip()
        elif self.summary_provider == "llama":
            messages = [
                {"role": "user", "content": prompt},
            ]
            outs = self.llama_pipe(messages, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, num_return_sequences=1)
            return outs[0]["generated_text"][-1]["content"].strip()
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
            outs = self.hf_pipe(messages, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, num_return_sequences=1)
            return outs[0]["generated_text"][-1]["content"].strip()

    def predict(self, query, source_text, augment_prompt=True, bertscore_threshold=0.85):
        if not augment_prompt:
            simple_prompt = query + "\n \n"
            simple_prompt += "--------------------------- \n"
            simple_prompt += "Besedilo: \n" + source_text + "\n \n" # mozda da dodam, u nastavku se nalaze odgovori na pitanja vezana za tekst
            simple_prompt += "--------------------------- \n"
            return self.generate_summary(simple_prompt, max_new_tokens=512)
        else:
            if self.question_generation_type == 'decomposition':
                questions = self.decomposer.decompose(query)
                answers = self.answer_decomposed_questions(questions, source_text, bertscore_threshold)
                augmented_prompt = query + "\n \n"
                augmented_prompt += "--------------------------- \n"
                augmented_prompt += "Besedilo: \n" + source_text + "\n \n" # mozda da dodam, u nastavku se nalaze odgovori na pitanja vezana za tekst
                augmented_prompt += "--------------------------- \n"
                augmented_prompt += "V nadaljevanju so odgovori na vprašanja, povezana z poizvedbo: \n"
                for q in questions:
                    a = answers.get(q, "Ni odgovora v tekstu.")
                    augmented_prompt += f"Vprašanje: {q}\nOdgovor: {a}\n"
                
                augmented_prompt += "\nNa podlagi zgornjih odgovorov, na kratko in jedrnato povzemite bistvo besedila v povezavi z začetno poizvedbo. "
                print("DEBUG - Augmented prompt for summary generation:"), 
                print(augmented_prompt)  # Debugging
                summary = self.generate_summary(augmented_prompt, max_new_tokens=512)
            elif self.question_generation_type == 'named entity from query':
                named_entities = self.ne_extractor.extract_entities(query)
                questions = []
                for ne in named_entities:
                    qg_prompt = self.build_qg_prompt(source_text, ne, self.qg_tokenizer)
                    q = self.generate_question(qg_prompt, max_length=48)
                    questions.append(q)

                print("DEBUG - Generated questions from named entities:", questions)  # Debugging
                answers = self.answer_decomposed_questions(questions, source_text, bertscore_threshold)
                augmented_prompt = query + "\n \n"
                augmented_prompt += "--------------------------- \n"
                augmented_prompt += "Besedilo: \n" + source_text + "\n \n" # mozda da dodam, u nastavku se nalaze odgovori na pitanja vezana za tekst
                augmented_prompt += "--------------------------- \n"
                augmented_prompt += "V nadaljevanju so odgovori na vprašanja, povezana z poizvedbo: \n"
                for q in questions:
                    a = answers.get(q, "Ni odgovora v tekstu.")
                    augmented_prompt += f"Vprašanje: {q}\nOdgovor: {a}\n"

                augmented_prompt += "\nNa podlagi zgornjih odgovorov, na kratko in jedrnato povzemite bistvo besedila v povezavi z začetno poizvedbo. "
                print("DEBUG - Augmented prompt for summary generation:"), 
                print(augmented_prompt)  # Debugging
                summary = self.generate_summary(augmented_prompt, max_new_tokens=512)

            return summary

    def answer_decomposed_questions(self, questions, source_text, bertscore_threshold : int = 0.85):
        answers = {}
        for q in questions:
            top_chunks = self.chunk_scorer.top_n_chunks(question=q, text=source_text, n=3)
            # filter only chunks where bertScore is greater than 0.85
            filtered_chunks = [chunk for chunk in top_chunks if chunk["f1"] > bertscore_threshold]
            if not filtered_chunks:
                answers[q] = "Ni odgovora v tekstu."
                continue

          # Get answers from QA model for each chunk
            chunk_answers = []
            for chunk in filtered_chunks:
                # Replace this with your actual QA model inference
                # Example: answer = self.qa_model.answer(question=q, context=chunk["chunk"])
                prompt = self.build_qa_prompt(q, chunk["chunk"])
                answer = self.generate_answer(prompt)
                answer = normalize_answer(answer)
                chunk_answers.append((answer, chunk["f1"]))

            # Find modal answer
            answer_counts = Counter([a for a, _ in chunk_answers])
            if not answer_counts:
                answers[q] = None
                continue

            most_common = answer_counts.most_common()
            max_count = most_common[0][1]
            candidates = [ans for ans, cnt in most_common if cnt == max_count]

            if len(candidates) == 1:
                final_answer = candidates[0]
            else:
                # Tie: pick answer from chunk with highest BertScore
                best = None
                best_score = -1
                for (ans, score) in chunk_answers:
                    if ans in candidates and score > best_score:
                        best = ans
                        best_score = score
                final_answer = best

            answers[q] = final_answer

        return answers

    def build_qa_prompt(self, question: str, context: str) -> str:
        p = (
            "Na podlagi podanega vprašanja in besedila generiraj samo en smiseln in pravilen odgovor, "
            "ki izhaja izključno iz konteksta tega besedila. "
            "Odgovor naj bo jasen, natančen in kratek (le nekaj besed). "
            "Če na vprašanje ni mogoče odgovoriti na podlagi predloženega besedila, ne ustvarite nobenega besedila."
            f"\n\nVprašanje: {question}\nBesedilo: {context}\nOdgovor:"
        )
        messages = [{"role": "user", "content": p}]
        return self.qa_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def build_qg_prompt(self, context: str, ans_text: str, tokenizer) -> str:
        if ans_text in context:
            context = context.replace(ans_text, f"<ans>{ans_text}</ans>", 1)
        user_content = (
            "Na podlagi naslednjega besedila in podanega odgovora generiraj samo eno vprašanje,"
            "na katerega je ta podani odgovor pravilen in smiseln izključno v kontekstu tega besedila."
            "Vprašanje naj bo oblikovano tako, da je prav podani odgovor (in ne katerikoli drug) edini pravilen odgovor."
            "Zaključi vprašanje z oznako [END]. "
            f"\nBesedilo: {context}\nOdgovor: {ans_text}\nVprašanje:"
        )
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def generate_answer(self, prompt: str, max_length: int = 256):
        inputs = self.qa_tokenizer(prompt, return_tensors="pt").to(self.qa_model.device)

        with torch.no_grad():
            out = self.qa_model.generate(
                **inputs,
                max_new_tokens=QA_MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                eos_token_id=self.qa_tokenizer.convert_tokens_to_ids("[END]"),        # hard stop
                top_p=TOP_P,
                use_cache=False,                 # critical for MC
                return_dict_in_generate=True,
                output_scores=True,
            )
        text = self.qa_tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        return text
    
    def generate_question(self, prompt: str, max_length: int = 64):
        inputs = self.qg_tokenizer(prompt, return_tensors="pt").to(self.qg_model.device)

        qg_stops = _encode_stop_sequences(self.qg_tokenizer, ["[END]"])
        stop_criteria = StoppingCriteriaList([StopOnSubsequence(qg_stops)]) if qg_stops else None

        with torch.no_grad():

            out = self.qg_model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,  # keep for speed
                return_dict_in_generate=True,
                output_scores=None,
                pad_token_id=self.qa_tokenizer.pad_token_id or self.qg_tokenizer.eos_token_id,
                stopping_criteria=stop_criteria,
            )

            prompt_len = inputs["input_ids"].shape[1]
            gen_id = out.sequences[0, prompt_len:]   # (B, T*)
            q_txt = self.qg_tokenizer.decode(gen_id, skip_special_tokens=True)
            return clean_question_text(q_txt)



            # out = self.qg_model.generate(
            #     **inputs,
            #     max_new_tokens=max_length,
            #     do_sample=False,
            #     temperature=1.0,
            #     eos_token_id=self.qg_tokenizer.convert_tokens_to_ids("[END]"),        # hard stop
            #     top_p=1.0,
            #     use_cache=True,                 # critical for MC
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     pad_token_id=self.qg_tokenizer.pad_token_id or self.qg_tokenizer.eos_token_id,
            #     stopping_criteria=stop_criteria,
            # )
        # text = self.qg_tokenizer.decode(out.sequences[0], skip_special_tokens=True)

NO_ANS_TOKEN = "<no_answer>"
END_SPLIT_RE   = re.compile(r"\[?\s*END\s*\]?", re.IGNORECASE)



def clean_answer_text(s: str) -> str:
    if "Odgovor:" in s:
        s = s.split("Odgovor:")[-1]
    s = END_TOK_RE.sub("", s)
    s = ROLE_PREFIX_RE.sub("", s)
    s = s.replace("\u200b", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()

def _strip_after_terminator(s: str) -> str:
    m = END_SPLIT_RE.search(s)
    return s[:m.start()] if m else s

def clean_question_text(s: str) -> str:
    s = s or ""
    s = ROLE_PREFIX_RE.sub("", s, count=1)
    s = _strip_after_terminator(s)
    s = s.replace("\u200b"," ").replace("\r"," ").replace("\n"," ")
    return " ".join(s.split())

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