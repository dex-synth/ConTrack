#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_root=${1:?Usage: bash scripts/h2r_ik_environments.sh /absolute/path/to/new-environments}
if [[ -e "$env_root" ]]; then
  echo "Refusing to replace an existing environment: $env_root" >&2
  exit 1
fi
uv venv --seed --python 3.10.21 "$env_root/mano"
# The unchanged chumpy source imports NumPy and pip during its build.
uv pip install --python "$env_root/mano/bin/python" numpy==1.23.5 setuptools==80.9.0
uv pip sync --python "$env_root/mano/bin/python" --no-build-isolation \
  "$root/h2r/mano-requirements.lock" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv venv --seed --python 3.11.16 "$env_root/ik"
uv pip sync --python "$env_root/ik/bin/python" "$root/h2r/ik-requirements.lock" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
echo "MANO: $env_root/mano/bin/python"
echo "IK:   $env_root/ik/bin/python"
