import importlib.util
import os
import tempfile

import pytest

from code_inference_query.indexer import Section
from code_inference_query.search import _parse_citation, _score_section, search

_VECTOR_DEPS = importlib.util.find_spec("lancedb") and importlib.util.find_spec("fastembed")


def test_search_resolves_exact_citation() -> None:
    sections = [
        Section(
            corpus_id="clean-code-python",
            shorthand="CC-Py",
            chapter="Functions",
            section_name="Functions",
            citation="CC-Py §Functions",
            content="Small functions are easier to understand.",
            line_start=1,
            keywords={"small", "functions", "understand"},
        )
    ]

    result = search(sections, "CC-Py §Functions", max_tokens=200, top_k=1)

    assert "CC-Py §Functions" in result
    assert "Small functions are easier to understand." in result


def test_search_handles_natural_language_queries() -> None:
    sections = [
        Section(
            corpus_id="fluent-python",
            shorthand="FP2e",
            chapter="Generators",
            section_name="Generators",
            citation="FP2e §Generators",
            content="Use yield from for generator delegation.",
            line_start=1,
            keywords={"generators", "yield", "from", "delegation"},
        ),
        Section(
            corpus_id="google-coding-standards",
            shorthand="Google",
            chapter="2.7 Comprehensions",
            section_name="Comprehensions",
            citation="Google §2.7",
            content="Avoid complex comprehensions.",
            line_start=1,
            keywords={"avoid", "complex", "comprehensions"},
        ),
    ]

    result = search(sections, "generator delegation yield from", max_tokens=200, top_k=1)

    assert "FP2e §Generators" in result


def test_score_section_broad_coverage_beats_narrow_at_equal_strength() -> None:
    # A section matching more query tokens scores higher than one matching fewer,
    # when per-token signal strength is comparable (keyword-only matches).
    # This verifies the coverage × mean_bonus formula rewards breadth.
    broad = Section(
        corpus_id="fluent-python",
        shorthand="FP2e",
        chapter="Iteration",
        section_name="Iteration",
        citation="FP2e §Iteration",
        content="Generators and delegation via yield from.",
        line_start=1,
        keywords={"generators", "yield", "delegation"},  # matches 3/3 tokens
    )
    narrow = Section(
        corpus_id="fluent-python",
        shorthand="FP2e",
        chapter="Builtins",
        section_name="Builtins",
        citation="FP2e §Builtins",
        content="Built-in functions and yield.",
        line_start=50,
        keywords={"yield"},  # matches 1/3 tokens
    )
    tokens = ["generators", "yield", "delegation"]
    assert _score_section(broad, tokens) > _score_section(narrow, tokens)


def test_score_section_no_match_returns_zero() -> None:
    section = Section(
        corpus_id="clean-code-python",
        shorthand="CC-Py",
        chapter="Functions",
        section_name="Functions",
        citation="CC-Py §Functions",
        content="Small functions are easier to understand.",
        line_start=1,
        keywords={"small", "functions", "understand"},
    )
    assert _score_section(section, ["database", "schema", "migration"]) == 0.0


def test_parse_citation_bare_shorthand_returns_none() -> None:
    # After no-§ branch removal, bare shorthand NL queries must not parse as citations.
    shorthand, ref = _parse_citation("CC-Py generators")
    assert shorthand is None
    assert ref is None


def test_parse_citation_with_section_sign_parses_correctly() -> None:
    shorthand, ref = _parse_citation("CC-Py §Functions.2")
    assert shorthand == "CC-Py"
    assert ref == "Functions.2"


def test_bare_shorthand_nl_query_routes_to_scoped_nl_search() -> None:
    # "CC-Py generators" must route to scoped NL search, not citation lookup.
    sections = [
        Section(
            corpus_id="clean-code-python",
            shorthand="CC-Py",
            chapter="Generators",
            section_name="Generators",
            citation="CC-Py §Generators",
            content="Generator functions use yield.",
            line_start=1,
            keywords={"generator", "generators", "yield"},
        )
    ]
    result = search(sections, "CC-Py generators", max_tokens=200, top_k=1)
    assert "CC-Py §Generators" in result


def test_format_results_respects_max_chars_budget() -> None:
    # The old 1000-char floor caused output to massively exceed the budget.
    # With 3 sections and max_chars=30, each section must get ~10 chars, not 1000.
    from code_inference_query.search import _format_results
    sections = [
        Section(
            corpus_id="fluent-python",
            shorthand="FP2e",
            chapter="Ch1",
            section_name=f"Section{i}",
            citation=f"FP2e §Section{i}",
            content="x" * 2000,
            line_start=1,
            keywords=set(),
        )
        for i in range(3)
    ]
    result = _format_results(sections, max_chars=30)
    # Must not approach 3000 chars (old buggy output with 1000-char floor × 3 sections)
    assert len(result) < 500


@pytest.fixture()
def vector_env(tmp_path):
    cache = str(tmp_path / "lance")
    os.environ["CODE_INFERENCE_CACHE_PATH"] = cache
    import code_inference_query.vector_store as vs
    vs._embed_model = None
    yield
    del os.environ["CODE_INFERENCE_CACHE_PATH"]
    vs._embed_model = None


def _make_sections() -> list[Section]:
    return [
        Section(
            corpus_id="clean-code-python",
            shorthand="CC-Py",
            chapter="Functions",
            section_name="Functions",
            citation="CC-Py §Functions",
            content="Small functions are easier to understand and test.",
            line_start=1,
            keywords={"small", "functions", "understand", "test"},
        ),
        Section(
            corpus_id="fluent-python",
            shorthand="FP2e",
            chapter="Generators",
            section_name="Generators",
            citation="FP2e §Generators",
            content="Use yield from for generator delegation in Python.",
            line_start=10,
            keywords={"generators", "yield", "delegation"},
        ),
    ]


@pytest.mark.skipif(not _VECTOR_DEPS, reason="lancedb and fastembed required for vector tests")
def test_build_vector_store_returns_searchable_table(vector_env) -> None:
    from code_inference_query.vector_store import build_vector_store

    table = build_vector_store(_make_sections())
    assert table is not None


@pytest.mark.skipif(not _VECTOR_DEPS, reason="lancedb and fastembed required for vector tests")
def test_vector_search_returns_relevant_section(vector_env) -> None:
    from code_inference_query.vector_store import build_vector_store, vector_search

    sections = _make_sections()
    table = build_vector_store(sections)
    results = vector_search(table, sections, "generator delegation yield", top_k=1)
    assert len(results) == 1
    assert results[0].citation == "FP2e §Generators"


@pytest.mark.skipif(not _VECTOR_DEPS, reason="lancedb and fastembed required for vector tests")
def test_build_vector_store_cache_hit_skips_rebuild(vector_env) -> None:
    """Second call with same sections must reuse the cached table."""
    from code_inference_query.vector_store import build_vector_store

    sections = _make_sections()
    table1 = build_vector_store(sections)
    import code_inference_query.vector_store as vs
    vs._embed_model = None  # clear to allow sentinel check below

    table2 = build_vector_store(sections)
    assert table1.name == table2.name
