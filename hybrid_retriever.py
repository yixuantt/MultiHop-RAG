"""Reusable Dense + BM25 + RRF + BGE reranker retrieval tool.

Create one HybridRetriever per process and reuse it for all queries.  The
module-level hybrid_retrieve() helper lazily creates one default instance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.postprocessor import FlagEmbeddingReranker
from llama_index.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.text_splitter import SentenceSplitter

from util import JSONReader


class CustomExtractor(BaseExtractor):
    """Keep the metadata fields used by the existing V3 retrieval scripts."""

    async def aextract(self, nodes) -> List[Dict[str, Any]]:
        return [
            {
                "title": node.metadata["title"],
                "source": node.metadata["source"],
                "published_at": node.metadata["published_at"],
            }
            for node in nodes
        ]


def tokenize(text: str) -> List[str]:
    """Tokenization used by bm25_retrieval.py."""
    return re.findall(r"[a-z0-9]+", text.lower())


def get_top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return top-k indices without sorting the full score array."""
    top_k = min(top_k, len(scores))
    if top_k <= 0:
        return np.array([], dtype=int)
    if top_k == len(scores):
        return np.argsort(scores)[::-1]

    candidates = np.argpartition(scores, -top_k)[-top_k:]
    return candidates[np.argsort(scores[candidates])[::-1]]


def normalize_text(text: str) -> str:
    """Use the same RRF document identity rule as rrf_fusion.py."""
    return " ".join(text.split())


class HybridRetriever:
    """In-memory hybrid retriever for arbitrary online queries.

    The corpus is split once per Python process.  A dense index is loaded from
    ``persist_dir`` when available; otherwise it is built once and persisted.
    BM25 and the BGE reranker remain in memory for subsequent calls.
    """

    def __init__(
        self,
        corpus_path: str = "dataset/corpus.json",
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = r"F:\model\bge-reranker-v2-m3",
        persist_dir: str = r"F:\storage\all-MiniLM-L6-v2_chunk256",
        chunk_size: int = 256,
        rrf_k: float = 60.0,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

        self.corpus_path = Path(corpus_path)
        self.persist_dir = Path(persist_dir)
        self.dense_model_name = dense_model_name
        self.reranker_model = reranker_model
        self.chunk_size = chunk_size
        self.rrf_k = rrf_k

        if not self.corpus_path.is_file():
            raise FileNotFoundError(f"Corpus file was not found: {self.corpus_path}")

        self.text_splitter = SentenceSplitter(chunk_size=self.chunk_size)
        self.embed_model = HuggingFaceEmbedding(
            model_name=dense_model_name,
            trust_remote_code=True,
        )
        # Retrieval does not generate text, so explicitly disable LLM usage.
        self.service_context = ServiceContext.from_defaults(
            llm=None,
            embed_model=self.embed_model,
            text_splitter=self.text_splitter,
        )

        self.nodes = self._load_corpus_nodes()
        self.node_texts = [self._node_search_text(node) for node in self.nodes]
        self.bm25 = BM25Okapi([tokenize(text) for text in tqdm(
            self.node_texts,
            desc="Tokenizing corpus for BM25",
        )])
        self.index = self._load_or_build_dense_index()
        # This constructor loads the cross-encoder once.  Only top_n changes
        # from one query to the next.
        self.reranker = FlagEmbeddingReranker(
            model=reranker_model,
            top_n=10,
        )

    def _load_corpus_nodes(self) -> List[Any]:
        reader = JSONReader()
        documents = reader.load_data(str(self.corpus_path))
        pipeline = IngestionPipeline(
            transformations=[self.text_splitter, CustomExtractor()]
        )
        print("Splitting corpus into nodes...")
        nodes = pipeline.run(documents=documents)
        if not nodes:
            raise ValueError(f"No nodes were created from: {self.corpus_path}")
        print(f"Loaded {len(nodes)} BM25 nodes.")
        return nodes

    def _load_or_build_dense_index(self):
        if self.persist_dir.is_dir() and any(self.persist_dir.iterdir()):
            print(f"Loading persisted dense index from: {self.persist_dir}")
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.persist_dir)
            )
            return load_index_from_storage(
                storage_context=storage_context,
                service_context=self.service_context,
            )

        print("No persisted dense index found. Building a new dense index...")
        index = VectorStoreIndex(
            self.nodes,
            service_context=self.service_context,
            show_progress=True,
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(self.persist_dir))
        print(f"Persisted dense index to: {self.persist_dir}")
        return index

    @staticmethod
    def _node_search_text(node: Any) -> str:
        """Metadata-inclusive text, matching V3 JSON retrieval outputs."""
        return node.get_content(metadata_mode=MetadataMode.LLM)

    def _node_key(self, node: Any) -> str:
        return normalize_text(self._node_search_text(node))

    def _dense_retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)

    def _bm25_retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        top_indices = get_top_k_indices(scores, top_k)
        return [
            NodeWithScore(node=self.nodes[index], score=float(scores[index]))
            for index in top_indices
        ]

    def _rrf_fuse(
        self,
        dense_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore],
        top_k: int,
    ) -> tuple[List[NodeWithScore], Dict[str, Dict[str, Any]]]:
        fused: Dict[str, Dict[str, Any]] = {}

        def add(nodes: List[NodeWithScore], source: str) -> None:
            for rank, scored_node in enumerate(nodes, start=1):
                key = self._node_key(scored_node.node)
                if key not in fused:
                    fused[key] = {
                        "node": scored_node.node,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "dense_score": None,
                        "bm25_rank": None,
                        "bm25_score": None,
                    }

                record = fused[key]
                record["rrf_score"] += 1.0 / (self.rrf_k + rank)
                record[f"{source}_rank"] = rank
                record[f"{source}_score"] = scored_node.get_score()

        add(dense_nodes, "dense")
        add(bm25_nodes, "bm25")

        ranked_records = sorted(
            fused.values(),
            key=lambda record: record["rrf_score"],
            reverse=True,
        )[:top_k]
        return (
            [
                NodeWithScore(
                    node=record["node"],
                    score=float(record["rrf_score"]),
                )
                for record in ranked_records
            ],
            fused,
        )

    def retrieve(
        self,
        query: str,
        retrieve_top_k: int = 10,
        rrf_top_k: int = 10,
        rerank_top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return documents ranked by Dense + BM25 + RRF + BGE reranking."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if min(retrieve_top_k, rrf_top_k, rerank_top_n) <= 0:
            raise ValueError("retrieve_top_k, rrf_top_k and rerank_top_n must be positive")

        print(
            "[1/4] Dense retrieval: encoding the query with "
            f"{self.dense_model_name} and searching {self.persist_dir}..."
        )
        dense_nodes = self._dense_retrieve(query, retrieve_top_k)
        print(f"      Dense returned {len(dense_nodes)} candidates.")

        print("[2/4] BM25 retrieval: scoring the query against the full corpus...")
        bm25_nodes = self._bm25_retrieve(query, retrieve_top_k)
        print(f"      BM25 returned {len(bm25_nodes)} candidates.")

        print("[3/4] RRF fusion: merging Dense and BM25 candidate rankings...")
        rrf_nodes, fused = self._rrf_fuse(
            dense_nodes=dense_nodes,
            bm25_nodes=bm25_nodes,
            top_k=rrf_top_k,
        )
        print(
            f"      RRF selected {len(rrf_nodes)} candidates "
            f"from {len(fused)} unique documents."
        )

        if not rrf_nodes:
            return []

        print(
            "[4/4] BGE reranking: scoring RRF candidates with "
            f"{self.reranker_model}..."
        )
        self.reranker.top_n = min(rerank_top_n, len(rrf_nodes))
        reranked_nodes = self.reranker.postprocess_nodes(
            rrf_nodes,
            query_bundle=QueryBundle(query_str=query),
        )
        print(f"      Reranker returned {len(reranked_nodes)} final documents.")

        documents: List[Dict[str, Any]] = []
        for rerank_rank, scored_node in enumerate(reranked_nodes, start=1):
            node = scored_node.node
            record = fused[self._node_key(node)]
            metadata = node.metadata or {}
            documents.append(
                {
                    "text": node.get_content(metadata_mode=MetadataMode.NONE),
                    "title": metadata.get("title"),
                    "source": metadata.get("source"),
                    "published_at": metadata.get("published_at"),
                    "dense_rank": record["dense_rank"],
                    "dense_score": record["dense_score"],
                    "bm25_rank": record["bm25_rank"],
                    "bm25_score": record["bm25_score"],
                    "rrf_score": record["rrf_score"],
                    "rerank_score": scored_node.get_score(),
                    "rerank_rank": rerank_rank,
                }
            )
        return documents


_default_retriever: Optional[HybridRetriever] = None


def configure_default_retriever(**kwargs: Any) -> HybridRetriever:
    """Create and store the process-wide retriever with custom settings."""
    global _default_retriever
    _default_retriever = HybridRetriever(**kwargs)
    return _default_retriever


def hybrid_retrieve(
    query: str,
    retrieve_top_k: int = 10,
    rrf_top_k: int = 10,
    rerank_top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Retrieve documents for an arbitrary query using the default tool instance."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = HybridRetriever()
    return _default_retriever.retrieve(
        query=query,
        retrieve_top_k=retrieve_top_k,
        rrf_top_k=rrf_top_k,
        rerank_top_n=rerank_top_n,
    )

def hybrid_retrieve_batch(
    queries: List[str],
    retrieve_top_k: int = 10,
    rrf_top_k: int = 10,
    rerank_top_n: int = 10,
) -> List[List[Dict[str, Any]]]:
    """Run the reusable tool for multiple queries in the same Python process."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = HybridRetriever()

    results = []
    for index, query in enumerate(queries, start=1):
        print(f"\n===== Query {index}/{len(queries)} =====")
        results.append(
            _default_retriever.retrieve(
                query=query,
                retrieve_top_k=retrieve_top_k,
                rrf_top_k=rrf_top_k,
                rerank_top_n=rerank_top_n,
            )
        )
    return results