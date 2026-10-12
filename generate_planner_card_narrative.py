"""LLM-generated content for the City Planner card (Chapter 6).

Kept for the existing commands and file references; the work is done by
generate_card_narrative.py, which writes planner_card_llm.json and
planner_card_llm_content_<provider>.tex as before.

    python generate_planner_card_narrative.py [--force] [--openai-model gpt-4.1]
"""

from __future__ import annotations

import sys

from generate_card_narrative import main

if __name__ == "__main__":
    main(["--stakeholder", "planner", *sys.argv[1:]])
