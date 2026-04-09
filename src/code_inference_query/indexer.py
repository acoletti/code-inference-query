"""Corpus loading, chunking, and section index builder."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .corpus_config import CORPUS_SPECS, SUBSECTION_BLOCKLIST, CorpusSpec


@dataclass
class Section:
    corpus_id: str
    shorthand: str
    chapter: str
    section_name: str
    citation: str
    content: str
    line_start: int
    keywords: set[str] = field(default_factory=set)


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keywords from text for search indexing."""
    words = re.findall(r"[a-zA-Z_]\w{2,}", text)
    return {w.lower() for w in words}


def _is_blocked_subsection(name: str) -> bool:
    stripped = name.strip()
    if stripped in SUBSECTION_BLOCKLIST:
        return True
    if len(stripped) < 5 or len(stripped) > 100:
        return True
    if stripped.startswith(("http", ">>>", "...", "#", "```")):
        return True
    # Skip lines that are all-caps (likely headers/navigation)
    if stripped.isupper() and len(stripped) > 3:
        return True
    return False


def _index_markdown_sections(
    spec: CorpusSpec, text: str
) -> list[Section]:
    """Index a markdown file by splitting on section/subsection headings."""
    lines = text.split("\n")
    sections = []
    current_chapter = ""
    current_section_name = ""
    current_lines = []
    current_start = 0
    subsection_count = 0

    def flush():
        if current_lines and current_section_name:
            content = "\n".join(current_lines).strip()
            if len(content) > 20:
                citation = f"{spec.shorthand} §{current_chapter}"
                if current_section_name != current_chapter:
                    citation = f"{spec.shorthand} §{current_section_name}"
                kw = _extract_keywords(content)
                kw.update(_extract_keywords(current_section_name))
                kw.update(_extract_keywords(current_chapter))
                sections.append(Section(
                    corpus_id=spec.corpus_id,
                    shorthand=spec.shorthand,
                    chapter=current_chapter,
                    section_name=current_section_name,
                    citation=citation,
                    content=content,
                    line_start=current_start,
                    keywords=kw,
                ))

    for i, line in enumerate(lines):
        # Check for section heading
        m = spec.section_pattern.match(line)
        if m:
            flush()
            groups = [g for g in m.groups() if g]
            if spec.corpus_id == "google-coding-standards":
                current_chapter = groups[0] if groups else ""
                current_section_name = groups[1] if len(groups) > 1 else groups[0]
            elif spec.corpus_id == "clean-code-inference":
                current_chapter = f"Ch{groups[0]}" if groups else ""
                current_section_name = groups[1].strip() if len(groups) > 1 else groups[0].strip()
            else:
                current_chapter = groups[0].strip() if groups else ""
                current_section_name = groups[0].strip()
            current_lines = [line]
            current_start = i
            subsection_count = 0
            continue

        # Check for subsection heading
        if spec.subsection_pattern:
            sm = spec.subsection_pattern.match(line)
            if sm and not _is_blocked_subsection(sm.group(1)):
                flush()
                subsection_count += 1
                sub_name = sm.group(1).strip()
                current_section_name = sub_name
                current_lines = [line]
                current_start = i
                continue

        current_lines.append(line)

    flush()
    return sections


def _index_google_sections(spec: CorpusSpec, text: str) -> list[Section]:
    """Google style guide has clean numbered sections — use them directly."""
    return _index_markdown_sections(spec, text)


def _index_design_patterns(spec: CorpusSpec, base_path: Path) -> list[Section]:
    """Index design pattern files from src/PatternName/Conceptual/main.py."""
    sections: list[Section] = []
    src_dir = base_path / spec.relative_path
    if not src_dir.exists():
        return sections

    for pattern_dir in sorted(src_dir.iterdir()):
        if not pattern_dir.is_dir() or pattern_dir.name.startswith("."):
            continue
        pattern_name = pattern_dir.name
        # Find main.py files (may be nested under Conceptual/, ThreadSafe/, etc.)
        for main_py in pattern_dir.rglob("main.py"):
            content = main_py.read_text(errors="replace")
            variant = main_py.parent.name
            section_name = pattern_name
            if variant not in ("Conceptual", pattern_name):
                section_name = f"{pattern_name}/{variant}"
            citation = f"DP-Py §{section_name}"
            sections.append(Section(
                corpus_id=spec.corpus_id,
                shorthand=spec.shorthand,
                chapter=pattern_name,
                section_name=section_name,
                citation=citation,
                content=content,
                line_start=0,
                keywords=_extract_keywords(content) | {pattern_name.lower()},
            ))
    return sections


def _index_example_code(spec: CorpusSpec, base_path: Path) -> list[Section]:
    """Index Fluent Python 2e example .py files by chapter."""
    sections: list[Section] = []
    ex_dir = base_path / spec.relative_path
    if not ex_dir.exists():
        return sections

    for chapter_dir in sorted(ex_dir.iterdir()):
        if not chapter_dir.is_dir() or chapter_dir.name.startswith("."):
            continue
        chapter_name = chapter_dir.name
        for py_file in sorted(chapter_dir.rglob("*.py")):
            content = py_file.read_text(errors="replace")
            if len(content.strip()) < 10:
                continue
            rel = py_file.relative_to(ex_dir)
            citation = f"FP2e-ex §{rel}"
            sections.append(Section(
                corpus_id=spec.corpus_id,
                shorthand=spec.shorthand,
                chapter=chapter_name,
                section_name=str(rel),
                citation=citation,
                content=content,
                line_start=0,
                keywords=_extract_keywords(content) | {chapter_name.lower()},
            ))
    return sections


def build_index(corpus_path: Path) -> list[Section]:
    """Build the full section index from all corpus files."""
    all_sections: list[Section] = []

    for spec in CORPUS_SPECS:
        if spec.is_directory:
            if spec.corpus_id == "design-patterns-python":
                all_sections.extend(_index_design_patterns(spec, corpus_path))
            elif spec.corpus_id == "example-code-2e":
                all_sections.extend(_index_example_code(spec, corpus_path))
            continue

        file_path = corpus_path / spec.relative_path
        if not file_path.exists():
            continue

        text = file_path.read_text(errors="replace")
        all_sections.extend(_index_markdown_sections(spec, text))

    return all_sections
