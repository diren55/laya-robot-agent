#!/usr/bin/env bash
# Install dependencies; all inference weights and robot assets are already bundled.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3.12}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
for name in vision sim motion; do
  if [ ! -x ".venvs/$name/bin/python" ]; then
    "$PYTHON" -m venv ".venvs/$name"
  fi
  ".venvs/$name/bin/python" -m pip install --upgrade pip wheel setuptools
  ".venvs/$name/bin/python" -m pip install -r "requirements/$name.txt"
done
.venvs/sim/bin/python -m pip install --no-deps -e vendor/robosuite
.venvs/motion/bin/python -m pip install --no-deps vendor/source_archives/source_0.zip vendor/source_archives/source_1.zip
printf '%s\n' 'Setup complete. Run: python3.12 run.py --seeds 1061201'
