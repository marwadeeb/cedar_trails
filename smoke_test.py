"""Quick end-to-end check: one question per persona, printing the full ReAct trace.

Usage:
    python smoke_test.py                       # Gemini (default)
    python smoke_test.py ollama qwen3:8b       # local model
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import cedar_trails as ct  # noqa: E402

model = {"provider": sys.argv[1], "model": sys.argv[2]} if len(sys.argv) == 3 else None
QUESTIONS = [
    ("rami", "An easy day hike in the Shouf in October for two beginners?"),
    ("hala", "We plan Sir ed-Danniyeh to Ehden in January with a fit group. Go or no-go?"),
    ("nour", "What should I pack for 3 days on the trail in the north in August?"),
]
for persona, question in QUESTIONS:
    r = ct.run_agent(question, persona=persona, model=model, log=False)
    print("=" * 70)
    print(f"{persona.upper()} | model: {r['cfg']['model']} | error: {r['error']} | tools: {r['tools_used']} | "
          f"calls: {r['llm_calls']} | format errors: {r['format_errors']} | {r['latency_s']} s")
    print(ct.format_trace(r))
