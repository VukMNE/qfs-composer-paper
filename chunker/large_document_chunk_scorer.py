from typing import List, Dict, Any

from bert_score import score as bertscore
import torch
import numpy as np


BERTSCORE_MODEL = "xlm-roberta-large"   # fallback: "xlm-roberta-base"

class LargeDocumentChunkScorer:
    def __init__(
        self,
        chunk_size_in_chars: int,
        bertscore_model: str = BERTSCORE_MODEL,
        lang: str = "sl",
        idf: bool = False,
        rescale_with_baseline: bool = False,
        batch_size: int = 16,
        device: str = None
    ):
        """
        Args:
          chunk_size_in_chars: fixed character window for chunking
          bertscore_model: BERTScore backbone (xlm-roberta-large works well for Slovene)
          lang: language code for BERTScore
          idf: whether to use IDF weighting (usually improves retrieval)
          rescale_with_baseline: BERTScore baseline rescaling
          batch_size: per-call BERTScore batch size
          device: "cuda" / "cpu" (auto if None)
        """
        self.chunk_size_in_chars = int(chunk_size_in_chars)
        self.bertscore_model = bertscore_model
        self.lang = lang
        self.idf = bool(idf)
        self.rescale_with_baseline = bool(rescale_with_baseline)
        self.batch_size = int(batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def top_n_chunks(self, question: str, text: str, n: int):
        """
        @:param question - the question
        @:param text - source text, original document that we want to analyze
        @:param n - number of highest ranked chunks to be returned

        this function returns n highest ranked chunks of text according to BertScore,
        which is measured by comparing similariy between the question and the chunk
        """

        chunks = self.split_text_into_chunks(text)
        if not chunks:
            return []

        # Build refs same size as chunks
        refs = [question] * len(chunks)

        # Compute BERTScore (vectorized)
        P, R, F1 = bertscore(
            cands=chunks,
            refs=refs,
            model_type=self.bertscore_model,
            lang=self.lang,
            idf=self.idf,
            rescale_with_baseline=self.rescale_with_baseline,
            batch_size=self.batch_size,
            device=self.device,
            verbose=False
        )
        f1_scores = F1.detach().cpu().numpy().astype("float64")

        # Rank by F1 descending
        n = min(int(n), len(chunks))
        order = np.argsort(-f1_scores)[:n]

        results: List[Dict[str, Any]] = []
        for idx in order:
            start = idx * self.chunk_size_in_chars
            end = min(start + self.chunk_size_in_chars, len(text))
            results.append({
                "index": int(idx),
                "chunk": chunks[idx],
                "start": int(start),
                "end": int(end),
                "f1": float(f1_scores[idx]),
            })
        return results


    def split_text_into_chunks(self, text: str):
        chunks = []
        processed_chars = 0

        while processed_chars < len(text):
            c = text[processed_chars:min(processed_chars + self.chunk_size_in_chars, len(text))]
            chunks.append(c)
            processed_chars += self.chunk_size_in_chars

        return chunks


