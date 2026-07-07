"""Pytest configuration for ensemble/scripts tests.

Adds the project root to sys.path so that the top-level `bin` package
(training code) is importable alongside the `ensemble` package.
"""
import sys
from pathlib import Path

# Project root: two levels up from this conftest (ensemble/scripts/ -> ensemble/ -> root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
