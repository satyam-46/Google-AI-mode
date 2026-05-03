"""Document chunking and semantic search utilities."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from collections import Counter
from typing import Iterable

from langchain_core.tools import StructuredTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional native dependency
    faiss = None

_LOG = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class SemanticSearchInput(BaseModel):
    query: str = Field(..., description="Question or phrase to search for.")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of chunks to return.")


class DocumentStore:
    """Thread-safe in-memory document store with optional Gemini embeddings + FAISS."""

    def __init__(self, embedding_model: str = "models/embedding-001") -> None:
        self._lock = asyncio.Lock()
        self._docs: list[Document] = []
        self._ids: set[str] = set()
        self._index = None
        self._embeddings = None

        if os.getenv("GOOGLE_API_KEY") and faiss is not None:
            try:
                self._embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            except Exception:
                _LOG.exception("Could not initialize Gemini embeddings; using lexical fallback")
                self._embeddings = None

    async def add_documents(self, docs: Iterable[Document]) -> None:
        """Chunk, optionally embed, and add documents to the store."""
        chunks = self._chunk_documents(list(docs))
        if not chunks:
            return

        async with self._lock:
            new_chunks = [chunk for chunk in chunks if chunk.id not in self._ids]
            if not new_chunks:
                return

            if self._embeddings is None or faiss is None:
                self._add_raw(new_chunks)
                return

            try:
                vectors = await self._embed_documents([chunk.text for chunk in new_chunks])
                self._add_vectors(new_chunks, vectors)
            except Exception:
                _LOG.exception("Embedding/FAISS add failed; storing chunks for lexical search")
                self._add_raw(new_chunks)

    async def add_search_results(self, results: Iterable[object]) -> None:
        """Add SearchResult-like objects to the store."""
        docs: list[Document] = []
        for index, result in enumerate(results):
            if isinstance(result, dict):
                url = str(result.get("url") or "")
                title = str(result.get("title") or "")
                content = str(result.get("content") or "")
            else:
                url = str(getattr(result, "url", ""))
                title = str(getattr(result, "title", ""))
                content = str(getattr(result, "content", ""))
            doc_id = url or f"search-result-{index}"
            docs.append(Document(id=doc_id, text=content, metadata={"url": url, "title": title}))
        await self.add_documents(docs)

    async def semantic_search(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the most relevant document chunks for a query."""
        async with self._lock:
            if self._embeddings is None or faiss is None or self._index is None:
                return self._lexical_search(query=query, top_k=top_k)

            try:
                query_vector = await self._embed_query(query)
                import numpy as np

                vector = np.array([query_vector], dtype="float32")
                _, indices = self._index.search(vector, top_k)
                return [self._docs[index] for index in indices[0] if 0 <= index < len(self._docs)]
            except Exception:
                _LOG.exception("FAISS search failed; using lexical fallback")
                return self._lexical_search(query=query, top_k=top_k)

    def as_tool(self) -> StructuredTool:
        async def _semantic_search_tool(query: str, top_k: int = 5) -> list[str]:
            return [doc.text for doc in await self.semantic_search(query=query, top_k=top_k)]

        return StructuredTool.from_function(
            coroutine=_semantic_search_tool,
            name="semantic_search",
            description="Search indexed evidence chunks for semantically relevant text.",
            args_schema=SemanticSearchInput,
        )

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks: list[Document] = []
        for doc in docs:
            for index, text in enumerate(splitter.split_text(doc.text)):
                chunks.append(Document(id=f"{doc.id}#{index}", text=text, metadata=doc.metadata))
        return chunks

    def _add_raw(self, chunks: list[Document]) -> None:
        for chunk in chunks:
            self._docs.append(chunk)
            self._ids.add(chunk.id)

    def _add_vectors(self, chunks: list[Document], vectors: list[list[float]]) -> None:
        import numpy as np

        matrix = np.array(vectors, dtype="float32")
        if self._index is None:
            self._index = faiss.IndexFlatL2(matrix.shape[1])
        self._index.add(matrix)
        self._add_raw(chunks)

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._embeddings is None:
            raise RuntimeError("Embeddings are not configured")
        return await self._embeddings.aembed_documents(texts)

    async def _embed_query(self, query: str) -> list[float]:
        if self._embeddings is None:
            raise RuntimeError("Embeddings are not configured")
        return await self._embeddings.aembed_query(query)

    def _lexical_search(self, query: str, top_k: int) -> list[Document]:
        query_terms = Counter(_TOKEN_RE.findall(query.lower()))
        if not query_terms:
            return self._docs[:top_k]

        scored = [(self._lexical_score(query_terms, doc.text), doc) for doc in self._docs]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]

    @staticmethod
    def _lexical_score(query_terms: Counter[str], text: str) -> float:
        doc_terms = Counter(_TOKEN_RE.findall(text.lower()))
        if not doc_terms:
            return 0.0

        overlap = sum(min(count, doc_terms[term]) for term, count in query_terms.items())
        coverage = overlap / sum(query_terms.values())
        density = overlap / math.sqrt(sum(doc_terms.values()))
        return coverage + density


__all__ = ["Document", "DocumentStore", "SemanticSearchInput"]
