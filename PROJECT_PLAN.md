# code-inference-query Distribution Plan

## Goal

Turn `code-inference-query` into a clean, distributable git repo with a reproducible local install path for Codex/Claude MCP use and a minimal test/build workflow.

## Phase 1: Repo Hygiene

- Initialize the directory as a git repo and add `.gitignore`.
- Keep packaging metadata in `pyproject.toml` explicit enough for local wheel builds.
- Document the supported install paths in `README.md`:
  - local venv install
  - `pipx` install
  - wheel/sdist build
  - MCP configuration examples

## Phase 2: Bootstrap Integration

- Ensure consumers like `multi_agent_codex` can bootstrap from:
  - a prebuilt executable
  - a local source repo path
- Keep the runtime entry point stable at `code-inference-query`.
- Prefer deterministic local installs into a tool-owned runtime directory over global user state.

## Phase 3: Quality Guardrails

- Maintain a lightweight test suite that exercises citation lookup and natural-language search behavior.
- Add a simple local test runner script.
- Add a local build script for wheel/sdist generation.

## Phase 4: Distribution Readiness

- Add remote git hosting once the local repo shape stabilizes.
- Add CI for:
  - unit tests
  - package build
  - smoke test of the console entry point
- Decide whether to publish to a package index or keep git/path installs as the supported model.

## Deferred Work

- Richer search relevance tests over fixture corpora
- release/versioning automation
- signed releases or package provenance
