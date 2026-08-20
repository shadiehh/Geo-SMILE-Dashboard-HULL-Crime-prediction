"""Load the canonical master prompt from repo-root llm/prompt.py.

Do not `from llm.prompt import ...` here. Streamlit runs dashboard/app.py
with `dashboard/` on sys.path, so that import would load this file again.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MASTER = Path(__file__).resolve().parents[2] / "llm" / "prompt.py"
_spec = importlib.util.spec_from_file_location("geosmile_master_prompt", _MASTER)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load master prompt from {_MASTER}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GEOSMILE_SYSTEM_PROMPT = _mod.GEOSMILE_SYSTEM_PROMPT
SYSTEM_PROMPT_VERSION = _mod.SYSTEM_PROMPT_VERSION
assert_prompt_semantics = _mod.assert_prompt_semantics

__all__ = [
    "GEOSMILE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT_VERSION",
    "assert_prompt_semantics",
]
