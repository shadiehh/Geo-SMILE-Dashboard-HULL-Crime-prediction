"""Canonical prompt package. Imports submodule by relative name only."""

from .prompt import (
    GEOSMILE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    assert_prompt_semantics,
)

__all__ = [
    "GEOSMILE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT_VERSION",
    "assert_prompt_semantics",
]
