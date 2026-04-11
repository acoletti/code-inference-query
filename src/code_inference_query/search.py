"""Query parsing, search, and response formatting."""

from __future__ import annotations

import re
from typing import Protocol

from .corpus_config import CORPUS_SPECS
from .indexer import Section


class SearchBackend(Protocol):
    def search(
        self,
        sections: list[Section],
        query: str,
        max_chars: int,
        top_k: int,
    ) -> str: ...

# Derive shorthands from the single source of truth in corpus_config
KNOWN_SHORTHANDS = {spec.shorthand for spec in CORPUS_SPECS}

# Build citation regex from shorthands — longest first so "FP2e-ex" matches before "FP2e"
_SHORTHAND_ALTERNATION = "|".join(
    sorted((re.escape(s) for s in KNOWN_SHORTHANDS), key=len, reverse=True)
)

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "that", "this", "what", "which",
    "who", "whom", "these", "those", "it", "its",
})


def _tokenize_query(q: str) -> list[str]:
    words = re.findall(r"[a-zA-Z_]\w{2,}", q)
    return [w.lower() for w in words if w.lower() not in STOPWORDS]


def _parse_citation(q: str) -> tuple[str | None, str | None]:
    """Try to parse a shorthand citation like 'CC-Py §Functions.2' or 'Google §2.7'."""
    # Match: SHORTHAND §SECTION
    m = re.match(rf"({_SHORTHAND_ALTERNATION})\s*§\s*(.+)", q.strip())
    if m:
        return m.group(1), m.group(2).strip()

    # Match: SHORTHAND SECTION (without §)
    m = re.match(rf"({_SHORTHAND_ALTERNATION})\s+(.+)", q.strip())
    if m:
        return m.group(1), m.group(2).strip()

    return None, None


def _score_section(section: Section, query_tokens: list[str]) -> float:
    """Score a section against query tokens using coverage × mean-bonus scoring.

    Rewards sections that match *most* query tokens (coverage) at high signal
    strength (mean_bonus), rather than penalizing longer queries.
    """
    if not query_tokens:
        return 0.0

    name_lower = section.section_name.lower()
    chapter_lower = section.chapter.lower()
    citation_lower = section.citation.lower()

    token_scores = []
    for token in query_tokens:
        t = 0.0
        if token in section.keywords:
            t += 1.0
        if token in name_lower:
            t += 2.0
        if token in chapter_lower:
            t += 1.5
        if token in citation_lower:
            t += 1.5
        token_scores.append(t)

    matched = [s for s in token_scores if s > 0]
    if not matched:
        return 0.0
    coverage = len(matched) / len(query_tokens)
    mean_bonus = sum(matched) / len(matched)
    return coverage * mean_bonus


def _truncate_content(content: str, max_chars: int = 6000) -> str:
    """Truncate content to approximately max_chars, breaking at line boundaries."""
    if len(content) <= max_chars:
        return content
    # Find the last newline before max_chars
    cut = content.rfind("\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    total_lines = content.count("\n")
    shown_lines = content[:cut].count("\n")
    return content[:cut] + f"\n\n[...truncated — showing {shown_lines}/{total_lines} lines]"


def search(
    sections: list[Section],
    query: str,
    max_tokens: int = 1500,
    top_k: int = 3,
    max_chars: int | None = None,
) -> str:
    """Search the corpus index and return formatted results."""
    if max_chars is None:
        max_chars = max_tokens * 4  # rough token-to-char ratio

    shorthand, section_ref = _parse_citation(query)

    if shorthand and section_ref:
        return _search_by_citation(sections, shorthand, section_ref, max_chars)

    # Check if query starts with a known shorthand (scoped natural language)
    first_word = query.strip().split()[0] if query.strip() else ""
    if first_word in KNOWN_SHORTHANDS:
        rest = query.strip()[len(first_word):].strip()
        return _search_natural_language(
            [s for s in sections if s.shorthand == first_word],
            rest, max_chars, top_k, scope=first_word,
        )

    return _search_natural_language(sections, query, max_chars, top_k)


def _search_by_citation(
    sections: list[Section],
    shorthand: str,
    section_ref: str,
    max_chars: int,
) -> str:
    """Resolve a specific citation like 'CC-Py §Functions.2' or 'Google §2.7'."""
    corpus_sections = [s for s in sections if s.shorthand == shorthand]
    if not corpus_sections:
        return f"No corpus found for shorthand '{shorthand}'."

    ref_lower = section_ref.lower().strip()

    # Exact citation match
    for s in corpus_sections:
        cit_suffix = s.citation.split("§", 1)[-1].strip().lower()
        if cit_suffix == ref_lower:
            return _format_result(s, max_chars)

    # For Google-style numeric refs (e.g., "2.7"), match section number prefix
    if re.match(r"\d+(\.\d+)*$", section_ref):
        matches = []
        for s in corpus_sections:
            if s.chapter.startswith(section_ref) or s.chapter == section_ref:
                matches.append(s)
        if matches:
            return _format_results(matches[:5], max_chars)

    # Numbered subsection ref like "Functions.2" → find 2nd subsection under "Functions"
    m = re.match(r"(.+?)\.(\d+)$", section_ref)
    if m:
        parent_name = m.group(1).lower()
        ordinal = int(m.group(2))
        parent_matches = [
            s for s in corpus_sections
            if parent_name in s.chapter.lower() or parent_name in s.section_name.lower()
        ]
        if ordinal <= len(parent_matches):
            return _format_result(parent_matches[ordinal - 1], max_chars)
        if parent_matches:
            return _format_results(parent_matches[:3], max_chars)

    # Fuzzy match on section name
    query_tokens = _tokenize_query(section_ref)
    scored = [(s, _score_section(s, query_tokens)) for s in corpus_sections]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, score in scored[:3] if score > 0]
    if top:
        return _format_results(top, max_chars)

    # List available sections as fallback
    available = sorted({s.citation for s in corpus_sections})[:20]
    return (
        f"No section matching '{section_ref}' found in {shorthand}.\n\n"
        f"Available sections (first 20):\n" + "\n".join(f"  - {c}" for c in available)
    )


def _search_natural_language(
    sections: list[Section],
    query: str,
    max_chars: int,
    top_k: int,
    scope: str | None = None,
) -> str:
    """Search by natural language query across all or scoped sections."""
    tokens = _tokenize_query(query)
    if not tokens:
        return "Query too short or all stopwords. Try a more specific query."

    scored = [(s, _score_section(s, tokens)) for s in sections]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, score in scored[:top_k] if score > 0]

    if not top:
        scope_msg = f" in {scope}" if scope else ""
        return f"No relevant sections found{scope_msg} for: {query}"

    return _format_results(top, max_chars)


def _format_result(section: Section, max_chars: int) -> str:
    """Format a single section result."""
    content = _truncate_content(section.content, max_chars)
    return (
        f"## {section.citation}\n"
        f"**Corpus**: {section.corpus_id} | **Chapter**: {section.chapter}\n"
        f"**Section**: {section.section_name}\n\n"
        f"{content}"
    )


def _format_results(sections: list[Section], max_chars: int) -> str:
    """Format multiple section results, dividing max_chars among them."""
    per_section_chars = max(max_chars // len(sections), 1000)
    parts = []
    for s in sections:
        content = _truncate_content(s.content, per_section_chars)
        parts.append(
            f"## {s.citation}\n"
            f"**Corpus**: {s.corpus_id} | **Chapter**: {s.chapter}\n"
            f"**Section**: {s.section_name}\n\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(parts)
