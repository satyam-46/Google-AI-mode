from typing import List, Optional
import asyncio
from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str


class DocumentStore:
    """In-memory document store placeholder guarded by an asyncio.Lock."""

    def __init__(self):
        self._docs = {}
        self._lock = asyncio.Lock()

    async def add(self, doc: Document) -> None:
        async with self._lock:
            self._docs[doc.id] = doc

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Document]:
        async with self._lock:
            return list(self._docs.values())[:top_k]
