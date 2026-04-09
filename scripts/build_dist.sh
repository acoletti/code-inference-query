#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

"$python_bin" -m pip install --upgrade build
"$python_bin" -m build "$repo_root"

echo "Built distribution artifacts in $repo_root/dist"
