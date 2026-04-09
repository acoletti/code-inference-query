"""MCP stdio server for code inference corpus queries."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .indexer import Section, build_index
from .search import _format_results, search
from .vector_store import build_vector_store, vector_search

# Resolve corpus path from env or default
_DEFAULT_CORPUS_PATH = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/code-inference"
)
CORPUS_PATH = Path(
    os.path.expanduser(
        os.environ.get("CODE_INFERENCE_CORPUS_PATH", _DEFAULT_CORPUS_PATH)
    )
)

mcp = FastMCP("code-inference-query")

# Module-level index — built on first query (lazy)
_index: list[Section] | None = None

# Sentinel distinguishing "not yet tried" (None) from "unavailable" after failure.
_VECTOR_STORE_UNAVAILABLE = object()
_vector_store = None  # None | _VECTOR_STORE_UNAVAILABLE | lancedb.Table


def _get_vector_store():
    """Return the lancedb Table, or None when vector extras are not installed."""
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = build_vector_store(_get_index())
        except Exception:
            _vector_store = _VECTOR_STORE_UNAVAILABLE
    if _vector_store is _VECTOR_STORE_UNAVAILABLE:
        return None
    return _vector_store


def _get_index() -> list[Section]:
    global _index
    if _index is None:
        if not CORPUS_PATH.exists():
            raise RuntimeError(
                f"Corpus path does not exist: {CORPUS_PATH}\n"
                f"Set CODE_INFERENCE_CORPUS_PATH to the correct location."
            )
        _index = build_index(CORPUS_PATH)
    return _index


@mcp.tool()
def query(
    q: str,
    max_tokens: int = 1500,
    top_k: int = 3,
) -> str:
    """Query the code inference corpus by shorthand citation or natural language.

    Examples:
      - "CC-Py §Functions.2" → Clean Code Python, Functions section, 2nd tip
      - "Google §2.7" → Google Style Guide section 2.7 (Comprehensions)
      - "FP2e §Generators" → Fluent Python 2e, generators chapter
      - "DP-Py §Strategy" → Design Patterns, Strategy pattern
      - "CC §Smells" → Clean Code, code smells chapter
      - "generator delegation yield from" → natural language search across all corpora

    Args:
        q: The query string — a shorthand citation (e.g. "CC-Py §Functions.2")
           or natural language (e.g. "decorator pattern python").
        max_tokens: Target response size in tokens (approximate). Default 1500.
        top_k: Number of sections to return for natural language queries. Default 3.

    Returns:
        Matching corpus excerpts formatted with citation headers.
    """
    index = _get_index()

    # Citation queries (contain §) go straight to keyword-based exact search.
    if "§" in q:
        return search(index, q, max_tokens=max_tokens, top_k=top_k)

    # Natural-language queries: try vector search, fall back to keyword search.
    try:
        table = _get_vector_store()
        if table is not None:
            nl_sections = vector_search(table, index, q, top_k)
            if nl_sections:
                return _format_results(nl_sections, max_tokens * 4)
    except Exception:
        pass

    return search(index, q, max_tokens=max_tokens, top_k=top_k)


@mcp.tool()
def list_corpora() -> str:
    """List all available corpora and their shorthands.

    Returns:
        A formatted table of corpus IDs, shorthands, and descriptions.
    """
    index = _get_index()
    # Collect unique corpora
    seen = {}
    for s in index:
        if s.corpus_id not in seen:
            seen[s.corpus_id] = s.shorthand
    from .corpus_config import CORPUS_SPECS
    lines = ["| Shorthand | Corpus | Sections |", "|-----------|--------|----------|"]
    for spec in CORPUS_SPECS:
        count = sum(1 for s in index if s.shorthand == spec.shorthand)
        lines.append(f"| `{spec.shorthand}` | {spec.description} | {count} |")
    lines.append(f"\n**Total sections indexed**: {len(index)}")
    lines.append(f"**Corpus path**: `{CORPUS_PATH}`")
    return "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
