import os
import time
from pathlib import Path
import numpy as np

# Prevent internal OpenAI client initialization even when using local models
os.environ["OPENAI_API_KEY"] = "not_used"

from raptor import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    BaseEmbeddingModel,
    BaseSummarizationModel,
    BaseQAModel,
)
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from hq_raptor import HybridQueryAdaptiveRetriever, estimate_query_specificity


class LocalEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text: str) -> list:
        return self.model.encode(text).tolist()


class LocalSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        max_len = min(max_tokens, 130)
        min_len = max(30, max_len // 2)
        try:
            return self.summarizer(
                context, max_length=max_len, min_length=min_len, do_sample=False
            )[0]["summary_text"]
        except Exception:
            return context[:500]


class LocalQAModel(BaseQAModel):
    def __init__(self, model_name="google/flan-t5-small"):
        self.qa_pipeline = pipeline("text2text-generation", model=model_name)

    def answer_question(self, context: str, question: str) -> str:
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        result = self.qa_pipeline(prompt, max_length=128, do_sample=False)
        return result[0]["generated_text"]


def keyword_recall(text: str, expected_keywords: list) -> float:
    text_lower = (text or "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return hits / max(len(expected_keywords), 1)


def get_base_context(ra: RetrievalAugmentation, question: str, top_k: int = 3) -> str:
    """Retrieve context from Base RAPTOR without triggering OpenAI-only defaults."""
    retriever = getattr(ra, "retriever", None)
    if retriever is not None and hasattr(retriever, "retrieve"):
        try:
            ctx = retriever.retrieve(question, top_k=top_k)
        except TypeError:
            ctx = retriever.retrieve(question)
        if isinstance(ctx, str):
            return ctx
        if isinstance(ctx, (list, tuple)):
            return "\n\n".join(
                getattr(item, "text", str(item)) for item in ctx
            )
    return ""


def main():
    emb_model = LocalEmbeddingModel()
    qa_model = LocalQAModel()
    config = RetrievalAugmentationConfig(
        embedding_model=emb_model,
        summarization_model=LocalSummarizationModel(),
        qa_model=qa_model,
    )

    tree_path = Path("demo/cinderella_local")
    if tree_path.exists():
        print(f"Loading existing local tree from {tree_path}...")
        ra = RetrievalAugmentation(config=config, tree=str(tree_path))
    else:
        print("Building tree from demo/sample.txt...")
        sample_path = (
            Path("demo/sample.txt") if Path("demo/sample.txt").exists() else Path("sample.txt")
        )
        ra = RetrievalAugmentation(config=config)
        ra.add_documents(sample_path.read_text(encoding="utf-8"))
        ra.save(str(tree_path))

    # Access internal tree object from RetrievalAugmentation
    tree_obj = getattr(ra, "tree", None) or getattr(ra.retriever, "tree", None)
    hq_retriever = HybridQueryAdaptiveRetriever(
        tree=tree_obj,
        embedding_model=emb_model,
        alpha=0.70,
        beta=0.20,
    )

    # Grimm's Cinderella ground-truth benchmark (matches demo/sample.txt)
    benchmark = [
        {
            "type": "Specific",
            "question": "What did Cinderella ask her father to bring back from the fair?",
            "keywords": ["twig", "hazel", "branch"],
        },
        {
            "type": "Specific",
            "question": "What was Cinderella's slipper made of in the story?",
            "keywords": ["gold", "golden", "silk"],
        },
        {
            "type": "Specific",
            "question": "What did the stepsisters cut off to make their feet fit the slipper?",
            "keywords": ["toe", "heel"],
        },
        {
            "type": "Broad",
            "question": "How did the birds help Cinderella throughout the story?",
            "keywords": ["lentils", "dress", "pigeons", "birds"],
        },
        {
            "type": "Broad",
            "question": "Summarize how the prince finally discovered Cinderella was the true bride.",
            "keywords": ["slipper", "shoe", "pigeons", "blood"],
        },
    ]

    base_ans_scores = {"Specific": [], "Broad": []}
    hq_ans_scores = {"Specific": [], "Broad": []}
    base_ctx_scores = {"Specific": [], "Broad": []}
    hq_ctx_scores = {"Specific": [], "Broad": []}
    base_times = []
    hq_times = []

    print("\n================ BENCHMARK RESULTS ================\n")
    for item in benchmark:
        q = item["question"]
        q_type = item["type"]
        kws = item["keywords"]

        # 1. Base RAPTOR
        t0 = time.perf_counter()
        base_context = get_base_context(ra, q, top_k=3)
        if base_context:
            base_ans = qa_model.answer_question(context=base_context, question=q)
        else:
            base_ans = ra.answer_question(question=q)
        base_ms = (time.perf_counter() - t0) * 1000.0
        base_ctx_rec = keyword_recall(base_context, kws) if base_context else 0.0
        base_ans_rec = keyword_recall(base_ans, kws)

        base_ans_scores[q_type].append(base_ans_rec)
        base_ctx_scores[q_type].append(base_ctx_rec)
        base_times.append(base_ms)

        # 2. HQ-RAPTOR
        t0 = time.perf_counter()
        top_nodes = hq_retriever.retrieve(q, top_k=3)
        hq_context = "\n\n".join(
            getattr(n, "text", str(n)) for n, _, _ in top_nodes
        )
        hq_ans = qa_model.answer_question(context=hq_context, question=q)
        hq_ms = (time.perf_counter() - t0) * 1000.0
        hq_ctx_rec = keyword_recall(hq_context, kws)
        hq_ans_rec = keyword_recall(hq_ans, kws)

        hq_ans_scores[q_type].append(hq_ans_rec)
        hq_ctx_scores[q_type].append(hq_ctx_rec)
        hq_times.append(hq_ms)

        layers_picked = [layer for _, _, layer in top_nodes]
        spec_val = estimate_query_specificity(q)

        print(f"[{q_type} | Specificity={spec_val:.2f}] Q: {q}")
        print(
            f"  Base RAPTOR ({base_ms:.1f} ms | CtxRec={base_ctx_rec:.2f} | "
            f"AnsRec={base_ans_rec:.2f}): {base_ans}"
        )
        print(
            f"  HQ-RAPTOR   ({hq_ms:.1f} ms | CtxRec={hq_ctx_rec:.2f} | "
            f"AnsRec={hq_ans_rec:.2f} | Layers={layers_picked}): {hq_ans}\n"
        )

    all_base_ctx = base_ctx_scores["Specific"] + base_ctx_scores["Broad"]
    all_hq_ctx = hq_ctx_scores["Specific"] + hq_ctx_scores["Broad"]

    print("================ SUMMARY TABLE FOR PAPER ================")
    print(
        f"Base RAPTOR | Specific AnsRec: {np.mean(base_ans_scores['Specific'])*100:.1f}% | "
        f"Broad AnsRec: {np.mean(base_ans_scores['Broad'])*100:.1f}% | "
        f"Avg CtxRec: {np.mean(all_base_ctx)*100:.1f}% | "
        f"Avg Latency: {np.mean(base_times):.1f} ms"
    )
    print(
        f"HQ-RAPTOR   | Specific AnsRec: {np.mean(hq_ans_scores['Specific'])*100:.1f}% | "
        f"Broad AnsRec: {np.mean(hq_ans_scores['Broad'])*100:.1f}% | "
        f"Avg CtxRec: {np.mean(all_hq_ctx)*100:.1f}% | "
        f"Avg Latency: {np.mean(hq_times):.1f} ms"
    )


if __name__ == "__main__":
    main()