from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K, get_qdrant_client)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # TODO: Implement Vietnamese word segmentation
    # 1. from underthesea import word_tokenize
    # 2. segmented = word_tokenize(text, format="text")
    # 3. return segmented.replace("_", " ")
    #
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    from underthesea import word_tokenize
    segmented = word_tokenize(text, format="text")
    # return segmented
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        # TODO: Implement BM25 indexing
        # 1. self.documents = chunks
        # 2. For each chunk: segment_vietnamese(chunk["text"]) → split by space
        # 3. self.corpus_tokens = [tokenized list for each chunk]
        # 4. from rank_bm25 import BM25Okapi
        #    self.bm25 = BM25Okapi(self.corpus_tokens)
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        # from rank_bm25 import BM25Okapi
        # self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None: return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [SearchResult(text=self.documents[i]["text"], score=scores[i], metadata=self.documents[i]["metadata"], method="bm25") for i in top_indices if scores[i] > 0]
        # TODO: Implement BM25 search
        # 1. if self.bm25 is None: return []
        # 2. tokenized_query = segment_vietnamese(query).split()
        # 3. scores = self.bm25.get_scores(tokenized_query)
        # 4. top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 5. Return [SearchResult(text=..., score=..., metadata=..., method="bm25")]
        #    Lọc scores[i] > 0 để bỏ docs không liên quan.


class DenseSearch:
    def __init__(self):
        self.client = get_qdrant_client()
        self._encoder = None
        self.documents: list[dict] = []
        self._vectors = None
        self._use_memory = False

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.documents = chunks
        texts = [c["text"] for c in chunks]
        self._vectors = self._get_encoder().encode(texts, show_progress_bar=True)

        try:
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = [PointStruct(id=i, vector=v.tolist(), payload={"doc_id": i}) for i, v in enumerate(self._vectors)]
            for start in range(0, len(points), 8):
                self.client.upsert(collection, points[start:start + 8], wait=True)
            self._use_memory = False
            print(f"  ✓ Qdrant indexed {len(points)} vectors", flush=True)
        except Exception as e:
            self._use_memory = True
            print(f"  ⚠️  Qdrant unavailable ({e}) — using in-memory dense search", flush=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self._use_memory or self._vectors is None:
            from numpy import dot
            from numpy.linalg import norm
            qv = self._get_encoder().encode(query)
            scores = [float(dot(qv, v) / (norm(qv) * norm(v) + 1e-9)) for v in self._vectors]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [
                SearchResult(
                    text=self.documents[i]["text"], score=scores[i],
                    metadata=self.documents[i].get("metadata", {}), method="dense",
                )
                for i in top_indices
            ]

        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(collection, query=query_vector, limit=top_k)
        results = []
        for pt in response.points:
            doc_id = pt.payload.get("doc_id", pt.id)
            if doc_id < len(self.documents):
                doc = self.documents[doc_id]
                results.append(SearchResult(
                    text=doc["text"], score=pt.score,
                    metadata=doc.get("metadata", {}), method="dense",
                ))
            elif "text" in pt.payload:
                results.append(SearchResult(
                    text=pt.payload["text"], score=pt.score,
                    metadata=pt.payload, method="dense",
                ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in merged
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
