"""tests/test_scenarios.py

End-to-end smoke test that replays the 6 required demo flows against the
LangGraph agent, all within a single shared conversation thread so that
requirement refinement and candidate references carry over correctly.

Run from the project root::

    python tests/test_scenarios.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matching_agent import chat  # noqa: E402

THREAD_ID = "test-scenarios"

FLOWS = [
    "Find candidates with React and 3+ years experience.",
    "Find candidates with Python, FastAPI, and 3+ years experience.",
    "Compare the top 3 candidates.",
    "Why did the top candidate rank higher than the second candidate?",
    "Make TypeScript a must-have and rerank.",
    "Generate interview questions for the top candidate.",
]


def main() -> None:
    for i, message in enumerate(FLOWS, start=1):
        print("=" * 80)
        print(f"Test {i}: {message}")
        print("=" * 80)
        reply = chat(message, thread_id=THREAD_ID)
        print(reply)
        print()


if __name__ == "__main__":
    main()
