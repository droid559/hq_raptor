import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class SimpleBM25:
    """Lightweight BM25 scorer over RAPTOR tree nodes with zero extra dependencies."""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens = docs_tokens
        self.n_docs = len(docs_tokens)
        self.doc_lens = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs > 0 else 1.0

        self.tf: List[Counter] = [Counter(d) for d in docs_tokens]
        self.df: Counter = Counter()
        for c in self.tf:
            for term in c:
                self.df[term] += 1

        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not query_tokens or self.n_docs == 0:
            return scores

        for i in range(self.n_docs):
            dl = self.doc_lens[i]
            tf_i = self.tf[i]
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-6)))
            s = 0.0
            for q in query_tokens:
                if q in tf_i and q in self.idf:
                    f = tf_i[q]
                    s += self.idf[q] * ((f * (self.k1 + 1.0)) / (f + denom_norm))
            scores[i] = s
        return scores


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def estimate_query_specificity(question: str) -> float:
    """
    Returns specificity in [0.0, 1.0]:
      1.0 -> very specific / factoid / entity query (prefers lower/leaf layers)
      0.0 -> broad / thematic / synthesis query (prefers higher summary layers)
    """
    q_lower = question.lower().strip()
    words = re.findall(r"\b\w+\b", question)
    n_words = len(words)

    broad_cues = [
        "summary", "summarize", "overall", "theme", "themes", "main idea",
        "how did", "why did", "throughout", "journey", "arc", "relationship",
        "compare", "evolution", "role of", "significance"
    ]
    specific_cues = [
        "who ", "what ", "when ", "where ", "which ", "how many", "name of", "exact"
    ]

    # Count capitalized words after position 0 as proxy for proper nouns/entities
    proper_nouns = sum(1 for w in words[1:] if w and w[0].isupper())

    score = 0.55
    if n_words <= 7:
        score += 0.20
    elif n_words >= 14:
        score -= 0.20

    if any(cue in q_lower for cue in broad_cues):
        score -= 0.30
    if any(q_lower.startswith(cue) or cue in q_lower for cue in specific_cues):
        score += 0.15
    if proper_nouns >= 1:
        score += 0.10

    return float(np.clip(score, 0.0, 1.0))


class HybridQueryAdaptiveRetriever:
    """
    Wraps a constructed RAPTOR tree to perform Hybrid (Dense + BM25)
    Query-Granularity Layer-Weighted Retrieval.
    """

    def __init__(
        self,
        tree: Any,
        embedding_model: Any,
        alpha: float = 0.70,
        beta: float = 0.20,
    ):
        self.tree = tree
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.beta = beta

        self.nodes: List[Any] = []
        self.layers: List[int] = []
        self.embeddings: np.ndarray = np.empty((0, 1), dtype=np.float32)
        self.bm25: SimpleBM25 = SimpleBM25([])
        self.max_layer: int = 1

        self._index_tree()

    def _extract_node_layer(self, node: Any, idx: int) -> int:
        # Adapt to common RAPTOR node attributes if present; fallback to 0
        for attr in ("layer", "level", "depth"):
            if hasattr(node, attr) and getattr(node, attr) is not None:
                return int(getattr(node, attr))
        if hasattr(self.tree, "layer_to_nodes") and isinstance(self.tree.layer_to_nodes, dict):
            for layer_id, layer_nodes in self.tree.layer_to_nodes.items():
                if node in layer_nodes:
                    return int(layer_id)
        return 0

    def _extract_node_text(self, node: Any) -> str:
        if isinstance(node, str):
            return node
        for attr in ("text", "content", "page_content"):
            if hasattr(node, attr) and getattr(node, attr) is not None:
                return str(getattr(node, attr))
        return str(node)

    def _extract_node_embedding(self, node: Any, text: str) -> np.ndarray:
        if hasattr(node, "embeddings") and isinstance(node.embeddings, dict) and node.embeddings:
            first_emb = next(iter(node.embeddings.values()))
            return np.asarray(first_emb, dtype=np.float32)
        if hasattr(node, "embedding") and getattr(node, "embedding") is not None:
            return np.asarray(getattr(node, "embedding"), dtype=np.float32)
        return np.asarray(self.embedding_model.create_embedding(text), dtype=np.float32)

    def _index_tree(self) -> None:
        raw_nodes: List[Any] = []
        if hasattr(self.tree, "all_nodes") and isinstance(self.tree.all_nodes, dict):
            raw_nodes = list(self.tree.all_nodes.values())
        elif hasattr(self.tree, "all_nodes") and isinstance(self.tree.all_nodes, list):
            raw_nodes = list(self.tree.all_nodes)
        elif hasattr(self.tree, "layer_to_nodes") and isinstance(self.tree.layer_to_nodes, dict):
            for layer_id in sorted(self.tree.layer_to_nodes.keys()):
                raw_nodes.extend(self.tree.layer_to_nodes[layer_id])

        texts: List[str] = []
        embs: List[np.ndarray] = []
        layers: List[int] = []

        for idx, node in enumerate(raw_nodes):
            text = self._extract_node_text(node)
            emb = self._extract_node_embedding(node, text)
            layer = self._extract_node_layer(node, idx)
            self.nodes.append(node)
            texts.append(text)
            embs.append(emb)
            layers.append(layer)

        self.layers = layers
        self.max_layer = max(layers) if layers else 1
        if embs:
            mat = np.vstack(embs).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            self.embeddings = mat / norms

        tokenized_docs = [_tokenize(t) for t in texts]
        self.bm25 = SimpleBM25(tokenized_docs)

    def _layer_weight(self, layer: int, specificity: float) -> float:
        norm_layer = layer / max(self.max_layer, 1)  # 0.0 (leaf) -> 1.0 (top summary)
        # High specificity favors low norm_layer; low specificity favors high norm_layer
        alignment = specificity * (1.0 - norm_layer) + (1.0 - specificity) * norm_layer
        return float(2.0 * alignment - 1.0)  # maps to [-1.0, +1.0]

    def retrieve(self, question: str, top_k: int = 5) -> List[Tuple[Any, float, int]]:
        if not self.nodes:
            return []

        q_emb = np.asarray(self.embedding_model.create_embedding(question), dtype=np.float32)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        dense_raw = self.embeddings @ q_emb
        dense_norm = _minmax_normalize(dense_raw)

        sparse_raw = self.bm25.score(_tokenize(question))
        sparse_norm = _minmax_normalize(sparse_raw)

        specificity = estimate_query_specificity(question)
        layer_bonus = np.array(
            [self._layer_weight(l, specificity) for l in self.layers],
            dtype=np.float32,
        )

        final_scores = (
            self.alpha * dense_norm
            + (1.0 - self.alpha) * sparse_norm
            + self.beta * layer_bonus
        )

        k = min(top_k, len(self.nodes))
        top_indices = np.argsort(-final_scores)[:k]

        return [
            (self.nodes[i], float(final_scores[i]), self.layers[i])
            for i in top_indices
        ]