# code-inference-query MCP Server

`code-inference-query` is a small Python MCP server that indexes the local code-inference corpus and returns relevant excerpts by shorthand citation or natural-language query. It is designed to keep large corpus files out of reviewer prompts and agent context windows.

## What This Repo Provides

- A stdio MCP server entry point: `code-inference-query`
- A lightweight local indexer over the corpus directory
- Citation lookup for shorthands like `CC-Py §Functions.2`
- Natural-language search across the configured corpora

## Repository Layout

```text
src/code_inference_query/
  server.py          MCP server entry point
  corpus_config.py   corpus manifest and shorthand definitions
  indexer.py         corpus loading and section indexing
  search.py          citation parsing and search formatting
  vector_store.py    optional vector search backend (requires [vector] extras)
scripts/
  install_local.sh   create a local venv and install the package
  build_dist.sh      build wheel and sdist artifacts
  test.sh            run the lightweight unit test suite
tests/
  test_search.py     search smoke tests (includes vector store tests)
```

## Prerequisites

- Python 3.11+
- Access to the corpus directory
- Network access if you need `pip` to fetch build/runtime dependencies on a fresh machine

## Git Repo Setup

If this directory is not already a git repo, initialize it before distributing changes:

```bash
cd /path/to/code-inference-query
git init -b main
```

Add a remote later when you are ready to publish it.

## Quick Start

### Local Virtualenv Install

```bash
cd /path/to/code-inference-query
bash ./scripts/install_local.sh

# verify the CLI
./.venv/bin/code-inference-query --help
```

Set `EDITABLE_INSTALL=1` if you want an editable install:

```bash
EDITABLE_INSTALL=1 bash ./scripts/install_local.sh
```

### Vector Search Extras

The default install uses keyword-based search. To enable the vector search backend (LanceDB + fastembed), install the `[vector]` optional extras:

```bash
pip install "/path/to/code-inference-query[vector]"
# or for the local venv:
./.venv/bin/pip install -e ".[vector]"
```

On first natural-language query the server will embed the corpus (~5–10 s) and cache the index at `~/.cache/code-inference-query/lance/`. Subsequent starts reuse the cache. The cache is invalidated automatically when the corpus changes.

### pipx Install

For a user-scoped install:

```bash
pipx install /path/to/code-inference-query
```

For active development:

```bash
pipx install -e /path/to/code-inference-query
```

## Build Distribution Artifacts

```bash
cd /path/to/code-inference-query
bash ./scripts/build_dist.sh
```

That produces wheel and sdist artifacts under `dist/`.

## Bootstrap Into Code Review Board

If you want `multi_agent_codex` to install this MCP into its managed Codex runtime, point the skill installer at this repo:

```bash
cd /path/to/multi_agent_codex
CODE_INFERENCE_QUERY_REPO_SOURCE=/path/to/code-inference-query \
CODE_INFERENCE_CORPUS_INSTALL_SOURCE=/path/to/code-inference \
./scripts/install_codex_skill.sh
```

That bootstrap flow creates a managed runtime under:

```text
${CODEX_HOME:-$HOME/.codex}/code-review/
```

and installs the executable at:

```text
${CODEX_HOME:-$HOME/.codex}/code-review/bin/code-inference-query
```

The bootstrap prefers installing from a modern `pyproject.toml` repo. For legacy source-only layouts, it falls back to copying `src/code_inference_query/` into the managed runtime and generating the console wrapper there.

## Run Tests

```bash
cd /path/to/code-inference-query
bash ./scripts/test.sh
```

## MCP Configuration

Example MCP config:

```json
{
  "mcpServers": {
    "code-inference-query": {
      "type": "stdio",
      "command": "/absolute/path/to/code-inference-query",
      "env": {
        "CODE_INFERENCE_CORPUS_PATH": "/absolute/path/to/code-inference"
      }
    }
  }
}
```

If you installed with the local venv script, the command path will usually be:

```text
/path/to/code-inference-query/.venv/bin/code-inference-query
```

If another tool bootstraps this package into a managed runtime directory, use the executable path that tool provides.

## Corpus Configuration

| Variable | Default | Purpose |
| ---------- | ------- | ------- |
| `CODE_INFERENCE_CORPUS_PATH` | `~/Library/Mobile Documents/com~apple~CloudDocs/code-inference` | Path to the corpus directory |
| `CODE_INFERENCE_CACHE_PATH` | `~/.cache/code-inference-query/lance` | Where the vector index is cached |
| `CODE_INFERENCE_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name for vector search |

## Query Examples

### Shorthand Citations

| Query | Returns |
|-------|---------|
| `CC-Py §Functions` | Clean Code Python functions section |
| `CC-Py §Functions.2` | 2nd subsection under Functions |
| `Google §2.7` | Google Style Guide section 2.7 |
| `FP2e §Generators` | Fluent Python generators chapter |
| `DP-Py §Strategy` | Strategy pattern source |
| `FP2e-ex §17-it-generator/sentence_gen.py` | Fluent Python example file |

### Natural Language

| Query | Returns |
|-------|---------|
| `generator delegation yield from` | best matching sections across corpora |
| `FP2e decorator closures` | scoped search within Fluent Python |
| `mutable default arguments` | matches across multiple corpora |

## Project Plan

The staged hardening plan for turning this into a cleaner distributable repo lives in [PROJECT_PLAN.md](/Users/amoscoletti/Library/Mobile%20Documents/com~apple~CloudDocs/code-inference-query/PROJECT_PLAN.md).
