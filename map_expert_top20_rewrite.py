"""Compatibility shim: use map_expert_top20.py."""

from map_expert_top20 import build_expert_top20 as build_custom_expert_top20
from map_expert_top20 import main

__all__ = ["build_custom_expert_top20", "main"]

if __name__ == "__main__":
    main()
