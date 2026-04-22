# conftest.py — mlgen3 root-level pytest configuration
"""
Root conftest: ensure the mlgen3 package (mlgen3/mlgen3/) resolves before the
orphan mlgen3/__init__.py that would otherwise shadow it when the matquant
repo root is on sys.path.
"""
import sys
from pathlib import Path

# This file lives at .../matquant/mlgen3/conftest.py
# parents[0] = .../matquant/mlgen3/
# parents[1] = .../matquant/
_MLGEN3_DIR = Path(__file__).resolve().parent      # .../matquant/mlgen3/
_REPO_ROOT   = _MLGEN3_DIR.parent                   # .../matquant/

for p in (_MLGEN3_DIR, _REPO_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
