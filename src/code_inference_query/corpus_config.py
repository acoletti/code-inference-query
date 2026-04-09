"""Corpus manifest — maps shorthand IDs to file paths and section-detection patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CorpusSpec:
    corpus_id: str
    shorthand: str
    relative_path: str
    section_pattern: re.Pattern
    subsection_pattern: re.Pattern | None = None
    description: str = ""
    is_directory: bool = False


# Section heading patterns per corpus
CORPUS_SPECS: list[CorpusSpec] = [
    CorpusSpec(
        corpus_id="clean-code-python",
        shorthand="CC-Py",
        relative_path="clean-code-python/README.md",
        section_pattern=re.compile(r"^## \*\*(.+?)\*\*\s*$"),
        subsection_pattern=re.compile(r"^### (.+)$"),
        description="Clean Code principles adapted to Python",
    ),
    CorpusSpec(
        corpus_id="fluent-python",
        shorthand="FP2e",
        relative_path="fluent_python/fluent_python.md",
        section_pattern=re.compile(r'^Part [IVX]+[.,] .(.+?).?\s*$|^(?:Chapter \d+[.:]\s*)(.+)$', re.IGNORECASE),
        subsection_pattern=re.compile(r"^([A-Z][A-Za-z][A-Za-z ,'':()\u2014\-&]{4,80})$"),
        description="Pythonic idioms from Fluent Python 2e",
    ),
    CorpusSpec(
        corpus_id="google-coding-standards",
        shorthand="Google",
        relative_path="google-coding-standards/google-coding-standards.md",
        section_pattern=re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$"),
        description="Google Python Style Guide",
    ),
    CorpusSpec(
        corpus_id="clean-code-inference",
        shorthand="CC",
        relative_path="clean-code-inference/clean_code_text.md",
        section_pattern=re.compile(r"^Chapter\s+(\d+)[:.]?\s*(.+)$"),
        subsection_pattern=re.compile(r"^([A-Z][A-Za-z][A-Za-z ,'':()—\-&!?]{4,80})$"),
        description="Clean Code (Robert C. Martin) — naming, functions, smells",
    ),
    CorpusSpec(
        corpus_id="design-patterns-python",
        shorthand="DP-Py",
        relative_path="design-patterns-python/src",
        description="Design patterns implemented in Python",
        section_pattern=re.compile(r".*"),  # unused — patterns loaded from directory
        is_directory=True,
    ),
    CorpusSpec(
        corpus_id="example-code-2e",
        shorthand="FP2e-ex",
        relative_path="example-code-2e",
        description="Runnable examples from Fluent Python 2e",
        section_pattern=re.compile(r".*"),  # unused — loaded from directory
        is_directory=True,
    ),
]

# False-positive subsection headings to skip in fluent_python and clean_code
SUBSECTION_BLOCKLIST = frozenset({
    "Skip to Content",
    "Settings",
    "Table of contents collapsed",
    "Table of Contents",
    "Note",
    "Tip",
    "Warning",
    "Image",
    "True",
    "False",
    "Click here to view code image",
    "Search for books, courses, events, and more",
    "Translations",
    "Introduction",
    "Requirements",
    "FAQ",
    "License",
    "Credits",
})
