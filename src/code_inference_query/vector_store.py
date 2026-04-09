"""Vector store: embed corpus sections and serve approximate nearest-neighbor queries.

Optional module — requires the [vector] extras (lancedb, fastembed).
Falls back gracefully when those packages are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .indexer import Section

# Corpora whose sections embed poorly under a natural-language model
# (raw Python source files rather than prose documentation).
_EXCLUDE_CORPUS_IDS = frozenset({"example-code-2e", "design-patterns-python"})

_DEFAULT_CACHE = Path.home() / ".cache" / "code-inference-query" / "lance"
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Module-level singleton — loaded once per process, reused for every query.
_embed_model = None


def _cache_path() -> Path:
    return Path(os.environ.get("CODE_INFERENCE_CACHE_PATH", str(_DEFAULT_CACHE)))


def _embed_model_name() -> str:
    return os.environ.get("CODE_INFERENCE_EMBED_MODEL", _DEFAULT_MODEL)


def _corpus_fingerprint(sections: list[Section], model_name: str) -> str:
    """SHA-256 of sorted (citation, line_start) pairs + model name."""
    pairs = sorted((s.citation, s.line_start) for s in sections)
    raw = json.dumps(pairs) + model_name
    return hashlib.sha256(raw.encode()).hexdigest()


def _markdown_sections_only(sections: list[Section]) -> list[Section]:
    """Exclude raw-source corpora that embed poorly under a prose model."""
    return [s for s in sections if s.corpus_id not in _EXCLUDE_CORPUS_IDS]


def _get_embed_model():
    """Return the module-level TextEmbedding singleton, loading on first call."""
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding  # noqa: PLC0415

        _embed_model = TextEmbedding(_embed_model_name())
    return _embed_model


def build_vector_store(sections: list[Section]):
    """Return a lancedb Table for the corpus, rebuilding only when the corpus changes.

    The table is cached at ``_cache_path()`` and validated via a SHA-256 fingerprint
    sidecar file.  A fingerprint match skips re-embedding entirely.

    Args:
        sections: Full flat index as returned by ``build_index()``.

    Returns:
        A ``lancedb.Table`` ready for ``.search(vector).limit(k).to_list()`` calls.
    """
    import lancedb  # noqa: PLC0415

    model_name = _embed_model_name()
    embed_sections = _markdown_sections_only(sections)
    fingerprint = _corpus_fingerprint(embed_sections, model_name)

    cache = _cache_path()
    fp_file = cache.parent / "fingerprint.txt"

    if fp_file.exists() and fp_file.read_text().strip() == fingerprint:
        db = lancedb.connect(str(cache))
        if "corpus" in db.list_tables().tables:
            return db.open_table("corpus")

    model = _get_embed_model()
    texts = [
        f"{s.citation} {s.section_name} {s.content[:512]}"
        for s in embed_sections
    ]
    embeddings = list(model.embed(texts))

    data = [
        {"citation": s.citation, "line_start": s.line_start, "vector": emb.tolist()}
        for s, emb in zip(embed_sections, embeddings)
    ]

    cache.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(cache))
    table = db.create_table("corpus", data, mode="overwrite")
    fp_file.write_text(fingerprint)
    return table


def vector_search(
    table,
    sections: list[Section],
    query: str,
    top_k: int,
) -> list[Section]:
    """Embed *query* and return the top-k nearest sections by cosine distance.

    Args:
        table: A ``lancedb.Table`` returned by ``build_vector_store()``.
        sections: Full flat index (used to resolve citation → Section).
        query: Natural-language query string.
        top_k: Number of results to return.

    Returns:
        Ordered list of matching ``Section`` objects (best match first).
    """
    model = _get_embed_model()
    query_vec = next(iter(model.embed([query]))).tolist()
    rows = table.search(query_vec).limit(top_k).to_list()

    by_citation = {s.citation: s for s in sections}
    return [by_citation[row["citation"]] for row in rows if row["citation"] in by_citation]
