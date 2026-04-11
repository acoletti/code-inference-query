"""Unit tests for the query() MCP tool in server.py."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

# Patch corpus path before importing so the module doesn't fail at import time
# when the real corpus directory is absent in CI / fresh checkouts.
import os
os.environ.setdefault(
    "CODE_INFERENCE_CORPUS_PATH",
    os.path.join(os.path.dirname(__file__), "fixtures", "corpus"),
)

from code_inference_query.server import (
    _CHARS_PER_TOKEN,
    _MAX_TOKENS_CEILING,
    query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_section(citation="CC §1", content="some content"):
    from code_inference_query.indexer import Section
    return Section(
        corpus_id="test",
        shorthand="CC",
        chapter="Ch1",
        section_name="Test Section",
        citation=citation,
        content=content,
        line_start=0,
        keywords={"test"},
    )


# ---------------------------------------------------------------------------
# max_tokens clamping
# ---------------------------------------------------------------------------

class TestMaxTokensClamp:
    """query() must clamp max_tokens to [1, _MAX_TOKENS_CEILING] before use."""

    def _captured_max_chars(self, max_tokens_input):
        """Return the max_chars value that reached search() for a citation query."""
        captured = {}
        with patch(
            "code_inference_query.server._get_index",
            return_value=[_make_section()],
        ):
            with patch(
                "code_inference_query.server.search",
                side_effect=lambda *a, **kw: captured.update(kw) or "",
            ):
                query("CC §1", max_tokens=max_tokens_input)
        return captured.get("max_chars")

    def test_clamp_zero(self):
        assert self._captured_max_chars(0) == 1 * _CHARS_PER_TOKEN

    def test_clamp_negative(self):
        assert self._captured_max_chars(-100) == 1 * _CHARS_PER_TOKEN

    def test_clamp_above_ceiling(self):
        assert self._captured_max_chars(99999) == _MAX_TOKENS_CEILING * _CHARS_PER_TOKEN

    def test_clamp_at_ceiling(self):
        assert self._captured_max_chars(_MAX_TOKENS_CEILING) == _MAX_TOKENS_CEILING * _CHARS_PER_TOKEN

    def test_clamp_at_one(self):
        assert self._captured_max_chars(1) == 1 * _CHARS_PER_TOKEN

    def test_typical_value_passes_through(self):
        assert self._captured_max_chars(1500) == 1500 * _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Citation routing (§ dispatch)
# ---------------------------------------------------------------------------

class TestCitationRouting:
    """Queries containing § must route to search() and never touch vector_search."""

    def test_citation_calls_search_not_vector(self):
        with patch("code_inference_query.server._get_index", return_value=[_make_section()]):
            with patch("code_inference_query.server.search", return_value="result") as mock_search:
                with patch("code_inference_query.server.vector_search") as mock_vec:
                    result = query("CC §Functions")
        mock_search.assert_called_once()
        mock_vec.assert_not_called()
        assert result == "result"

    def test_citation_passes_max_chars_not_max_tokens(self):
        """search() must receive max_chars= (pre-computed), not max_tokens=."""
        captured = {}
        with patch("code_inference_query.server._get_index", return_value=[_make_section()]):
            with patch(
                "code_inference_query.server.search",
                side_effect=lambda *a, **kw: captured.update(kw) or "",
            ):
                query("CC §1", max_tokens=500)
        assert "max_chars" in captured
        assert captured["max_chars"] == 500 * _CHARS_PER_TOKEN
        assert "max_tokens" not in captured


# ---------------------------------------------------------------------------
# Natural-language vector path
# ---------------------------------------------------------------------------

class TestNaturalLanguageRouting:
    """NL queries should attempt vector search, then fall back to keyword search."""

    def test_nl_uses_vector_when_available(self):
        mock_table = MagicMock()
        section = _make_section()
        with patch("code_inference_query.server._get_index", return_value=[section]):
            with patch("code_inference_query.server._get_vector_store", return_value=mock_table):
                with patch(
                    "code_inference_query.server.vector_search",
                    return_value=[section],
                ):
                    with patch(
                        "code_inference_query.server._format_results",
                        return_value="vector result",
                    ) as mock_fmt:
                        result = query("generator delegation")
        mock_fmt.assert_called_once()
        assert result == "vector result"

    def test_nl_fallback_when_no_vector_store(self):
        with patch("code_inference_query.server._get_index", return_value=[_make_section()]):
            with patch("code_inference_query.server._get_vector_store", return_value=None):
                with patch(
                    "code_inference_query.server.search", return_value="keyword result"
                ) as mock_search:
                    result = query("generator delegation")
        mock_search.assert_called_once()
        assert result == "keyword result"

    def test_nl_fallback_when_vector_returns_empty(self):
        mock_table = MagicMock()
        with patch("code_inference_query.server._get_index", return_value=[_make_section()]):
            with patch("code_inference_query.server._get_vector_store", return_value=mock_table):
                with patch("code_inference_query.server.vector_search", return_value=[]):
                    with patch(
                        "code_inference_query.server.search", return_value="keyword result"
                    ) as mock_search:
                        result = query("generator delegation")
        mock_search.assert_called_once()
        assert result == "keyword result"

    def test_nl_fallback_on_vector_exception_calls_search(self):
        mock_table = MagicMock()
        with patch("code_inference_query.server._get_index", return_value=[_make_section()]):
            with patch("code_inference_query.server._get_vector_store", return_value=mock_table):
                with patch(
                    "code_inference_query.server.vector_search",
                    side_effect=RuntimeError("lancedb exploded"),
                ):
                    with patch(
                        "code_inference_query.server.search", return_value="keyword result"
                    ) as mock_search:
                        result = query("generator delegation")
        mock_search.assert_called_once()
        assert result == "keyword result"

    def test_nl_fallback_on_vector_exception_logs_error(self):
        mock_table = MagicMock()
        with patch("code_inference_query.server._get_index", return_value=[_make_section()]):
            with patch("code_inference_query.server._get_vector_store", return_value=mock_table):
                with patch(
                    "code_inference_query.server.vector_search",
                    side_effect=RuntimeError("lancedb exploded"),
                ):
                    with patch("code_inference_query.server.search", return_value=""):
                        with patch("code_inference_query.server.logger") as mock_logger:
                            query("generator delegation")
        mock_logger.error.assert_called_once()
        call_kwargs = mock_logger.error.call_args
        assert call_kwargs.kwargs.get("exc_info") is True

    def test_nl_vector_receives_precomputed_max_chars(self):
        """_format_results must receive max_chars = max_tokens * _CHARS_PER_TOKEN."""
        mock_table = MagicMock()
        section = _make_section()
        captured = {}
        with patch("code_inference_query.server._get_index", return_value=[section]):
            with patch("code_inference_query.server._get_vector_store", return_value=mock_table):
                with patch("code_inference_query.server.vector_search", return_value=[section]):
                    with patch(
                        "code_inference_query.server._format_results",
                        side_effect=lambda secs, mc: captured.update({"max_chars": mc}) or "",
                    ):
                        query("factory pattern", max_tokens=800)
        assert captured["max_chars"] == 800 * _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Characterization: § sign in NL queries misroutes (known heuristic limitation)
# ---------------------------------------------------------------------------

class TestSectionSignHeuristic:
    """Document the known behaviour: any query containing § routes to citation search.

    This is a characterization test — it documents existing behaviour, not a bug fix.
    A NL query like 'what does § mean?' will silently use citation routing.
    """

    def test_nl_query_with_section_sign_routes_to_search_not_vector(self):
        with patch("code_inference_query.server._get_index", return_value=[_make_section()]):
            with patch("code_inference_query.server.search", return_value="") as mock_search:
                with patch("code_inference_query.server.vector_search") as mock_vec:
                    query("what does § mean in python?")
        mock_search.assert_called_once()
        mock_vec.assert_not_called()
