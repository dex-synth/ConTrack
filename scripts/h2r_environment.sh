#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_path=${1:?Usage: bash scripts/h2r_environment.sh /absolute/path/to/environment}
if [[ -e "$env_path" ]]; then
  echo "Refusing to replace an existing environment: $env_path" >&2
  exit 1
fi
uv venv --python 3.12 "$env_path"
uv pip sync --python "$env_path/bin/python" "$root/h2r/requirements.lock" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
echo "Run from $root with $env_path/bin/python -m contrack_h2r.prepare --help"
