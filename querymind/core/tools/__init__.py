"""LangChain-compatible tools."""

from core.tools.document_store import Document, DocumentStore
from core.tools.web_search import SearchResult, web_search, web_search_tool

__all__ = ["Document", "DocumentStore", "SearchResult", "web_search", "web_search_tool"]
