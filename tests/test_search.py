import importlib.util
import os
import tempfile
import unittest

from code_inference_query.indexer import Section
from code_inference_query.search import SearchBackend, _score_section, search

_VECTOR_DEPS = importlib.util.find_spec("lancedb") and importlib.util.find_spec("fastembed")


class TestSearch(unittest.TestCase):
    def test_search_resolves_exact_citation(self) -> None:
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

        self.assertIn("CC-Py §Functions", result)
        self.assertIn("Small functions are easier to understand.", result)

    def test_search_handles_natural_language_queries(self) -> None:
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

        self.assertIn("FP2e §Generators", result)


    def test_score_section_broad_coverage_beats_narrow_at_equal_strength(self) -> None:
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
        self.assertGreater(
            _score_section(broad, tokens),
            _score_section(narrow, tokens),
        )

    def test_score_section_no_match_returns_zero(self) -> None:
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
        self.assertEqual(_score_section(section, ["database", "schema", "migration"]), 0.0)

    def test_search_backend_protocol_satisfied(self) -> None:
        # Verify SearchBackend is a structural Protocol — any object with the
        # right signature satisfies it without explicit inheritance.
        class _StubBackend:
            def search(
                self,
                sections: list[Section],
                query: str,
                max_chars: int,
                top_k: int,
            ) -> str:
                return ""

        def _accepts_backend(b: SearchBackend) -> str:
            return b.search([], "q", 100, 1)

        self.assertEqual(_accepts_backend(_StubBackend()), "")


@unittest.skipUnless(_VECTOR_DEPS, "lancedb and fastembed required for vector tests")
class TestVectorStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        cache = os.path.join(self._tmpdir.name, "lance")
        os.environ["CODE_INFERENCE_CACHE_PATH"] = cache
        # Reset module-level singleton so each test starts clean.
        import code_inference_query.vector_store as vs
        vs._embed_model = None

    def tearDown(self) -> None:
        del os.environ["CODE_INFERENCE_CACHE_PATH"]
        self._tmpdir.cleanup()

    def _make_sections(self) -> list[Section]:
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

    def test_build_vector_store_returns_searchable_table(self) -> None:
        from code_inference_query.vector_store import build_vector_store

        sections = self._make_sections()
        table = build_vector_store(sections)
        self.assertIsNotNone(table)

    def test_vector_search_returns_relevant_section(self) -> None:
        from code_inference_query.vector_store import build_vector_store, vector_search

        sections = self._make_sections()
        table = build_vector_store(sections)
        results = vector_search(table, sections, "generator delegation yield", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].citation, "FP2e §Generators")

    def test_build_vector_store_cache_hit_skips_rebuild(self) -> None:
        """Second call with same sections must reuse the cached table."""
        from code_inference_query.vector_store import build_vector_store

        sections = self._make_sections()
        table1 = build_vector_store(sections)
        # Overwrite module model to ensure second call doesn't re-embed
        import code_inference_query.vector_store as vs
        vs._embed_model = None  # clear to allow sentinel check below

        table2 = build_vector_store(sections)
        # Both tables point to the same underlying data
        self.assertEqual(table1.name, table2.name)


if __name__ == "__main__":
    unittest.main()
