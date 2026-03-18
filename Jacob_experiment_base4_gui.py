"""
Jacob_experiment_base4_gui.py — Thin compatibility shim.

All logic has been consolidated into experiment_core.py.
This file re-exports ExperimentRunner so that existing imports continue to work.
"""
import os
import sys

# Ensure experiment_core.py (in the same directory) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_core import (  # noqa: F401  (re-exported for backward compat)
    ExperimentRunner,
    STRATEGY_REGISTRY,
    DEFAULT_STRATEGY_KEYS,
)
from subplot_registry import (  # noqa: F401
    SUBPLOT_REGISTRY,
    DEFAULT_SUBPLOT_IDS,
)
