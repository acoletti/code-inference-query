#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${VENV_DIR:-$repo_root/.venv}"
python_bin="${PYTHON_BIN:-python3}"
editable="${EDITABLE_INSTALL:-0}"

"$python_bin" -m venv "$venv_dir"

if [ "$editable" = "1" ]; then
  "$venv_dir/bin/python" -m pip install -e "$repo_root"
else
  "$venv_dir/bin/python" -m pip install "$repo_root"
fi

echo "Installed code-inference-query into $venv_dir"
echo "CLI available at $venv_dir/bin/code-inference-query"
