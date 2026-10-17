"""Persona, prompt-engineering and LLM-configuration experiments for Cedar Trails.

Usage:
    python experiments.py --dry-run      # list planned runs
    python experiments.py                # run everything (resumable; one thread per model)
    python experiments.py --only E3      # one experiment
    python experiments.py --judge        # LLM-as-judge scoring of finished runs (Gemini)
"""
import argparse
import json
import re
import threading
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import cedar_trails as ct  # noqa: E402

RESULTS = ct.BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)
RUNS_FILE = RESULTS / "runs.jsonl"
JUDGE_FILE = RESULTS / "judge.jsonl"

MODELS = {
    "gemini-3.5-flash-lite": {"provider": "gemini", "model": ct.GEMINI_MODEL},
    "qwen3.8-27b": {"provider": "groq", "model": "qwen/qwen3.8-27b"},
    "gpt-oss-120b": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "gpt-oss-20b": {"provider": "groq", "model": "openai/gpt-oss-20b"},
}

# Test cases: each persona gets two tasks with checkable expectations.
# `facts`: every inner list must have at least one alternative present in the final answer.
CASES = {
    "R1": {"persona": "rami", "question": "An easy day hike in the Shouf in October for two beginners?",
           "tools": {"search_trail_sections", "estimate_hike_time"},
           "facts": [["S08", "Falougha"], ["4.5"]]},
    "R2": {"persona": "rami", "question": "I'm fit and want a challenging ridge walk through cedars in July, under 18 km. Ideas?",
           "tools": {"search_trail_sections", "estimate_hike_time"},
           "facts": [["S09", "Barouk"], ["4.9", "4.1"]]},
    "H1": {"persona": "hala", "question": "We plan Sir ed-Danniyeh to Ehden in January with a fit group. Go or no-go?",
           "tools": {"check_safety_alerts", "get_season_conditions"}, "verdict": "NO-GO",
           "facts": [["NO-GO"], ["danger", "impassable"], ["high"]]},
    "H2": {"persona": "hala", "question": "Is Baskinta to Zaarour (S07) OK for a family hike in May?",
           "tools": {"check_safety_alerts", "get_season_conditions"}, "verdict": "GO",
           "facts": [["9", "9.0"], ["low"]]},
    "N1": {"persona": "nour", "question": "What should I pack for 3 days on the trail in the north in August?",
           "tools": {"build_packing_list"}, "facts": [["3.5"], ["8.4"], ["electrolyte"]]},
    "N2": {"persona": "nour", "question": "Packing list for a single winter day hike in Mount Lebanon in February?",
           "tools": {"build_packing_list"}, "facts": [["2.5"], ["8.0", "8 kg", "~8"], ["crampons", "microspikes"]]},
}
FORMAT_CHECKS = {
    "rami": [r"\*\*Top pick:\*\*", r"\*\*Stats:\*\*"],
    "hala": [r"VERDICT:\s*(GO WITH CAUTION|NO-GO|GO)", r"REASONS:"],
    "nour": [r"###\s*Essentials", r"\*\*Water:\*\*"],
}

PROMPT_VARIANTS = [{"cot": c, "few_shot": f, "placement": p}
                   for c in (True, False) for f in (True, False) for p in ("system", "message")]


def plan():
    """Every planned run as a dict with a stable run_id."""
    runs = []

    def add(exp, model, case, override=None, variant=None, rep=0, label=""):
        # E1-E4 ran before the budget-warning fix existed, so it is off unless an experiment sets it.
        variant = {"cot": True, "few_shot": True, "placement": "system", "instructions": "full", "budget_warning": False, **(variant or {})}
        override = override or {}
        rid = f"{exp}|{model}|{case}|{label or 'base'}|r{rep}"
        runs.append({"run_id": rid, "exp": exp, "model_key": model, "case": case, "label": label or "base",
                     "override": override, "variant": variant, "rep": rep})

    # E1: every persona on its own tasks across four models.
    for m in MODELS:
        for case in CASES:
            add("E1", m, case)
    # E2: persona-config fit. Each persona runs with each persona's LLM config.
    for case in ("R1", "H1", "N1", "R2", "H2", "N2"):
        for cfg_owner in ct.PERSONAS:
            add("E2", "gpt-oss-120b", case, override=ct.PERSONAS[cfg_owner]["config"], label=f"cfg_{cfg_owner}")
    # E3: prompt engineering - CoT x few-shot x system/in-message placement.
    for case in ("R1", "H1", "N1"):
        for v in PROMPT_VARIANTS:
            label = f"{'cot' if v['cot'] else 'nocot'}_{'few' if v['few_shot'] else 'zero'}_{v['placement']}"
            add("E3", "gpt-oss-20b", case, variant=v, label=label)
    # E4: single-parameter sweeps on Rami's task (3 repeats to measure output diversity).
    sweeps = {"temp_0.0": {"temperature": 0.0}, "temp_0.4": {"temperature": 0.4}, "temp_0.8": {"temperature": 0.8},
              "temp_1.2": {"temperature": 1.2}, "temp_1.6": {"temperature": 1.6}, "top_p_0.3": {"top_p": 0.3},
              "presence_1.5": {"presence_penalty": 1.5}, "frequency_1.5": {"frequency_penalty": 1.5},
              "max_tokens_150": {"max_tokens": 150}}
    for label, override in sweeps.items():
        for rep in range(3):
            add("E4", "qwen3.8-27b", "R1", override=override, label=label, rep=rep)
    # E4b: top_k only exists on Gemini.
    for label, override in {"top_k_1": {"top_k": 1}, "top_k_64": {"top_k": 64}, "top_k_400": {"top_k": 400}}.items():
        for rep in range(3):
            add("E4", "gemini-3.5-flash-lite", "R1", override=override, label=label, rep=rep)
    # E5a: instructions ablation - full instructions block vs a minimal one (tools + action syntax only).
    for case in CASES:
        for mode in ("full", "minimal"):
            add("E5", "qwen3.8-27b", case, variant={"instructions": mode}, label=f"instr_{mode}")
    # E5b: last-step budget warning, on the cases that ran out of steps in E1.
    for model in ("qwen3.8-27b", "gpt-oss-20b"):
        for case in ("R2", "H2"):
            for warn in (False, True):
                for rep in range(2):
                    add("E5", model, case, variant={"budget_warning": warn}, label=f"budget_{'on' if warn else 'off'}", rep=rep)
    return runs


def score(case_id, result):
    """Automatic metrics for one run."""
    case = CASES[case_id]
    answer = result["final_answer"] or ""
    used = set(result["tools_used"])
    fmt = all(re.search(p, answer) for p in FORMAT_CHECKS[case["persona"]])
    facts = [any(alt.lower() in answer.lower() for alt in alts) for alts in case["facts"]]
    metrics = {
        "completed": result["error"] is None and not answer.startswith(("I couldn't", "Sorry")),
        "format_ok": bool(fmt),
        "tool_recall": len(case["tools"] & used) / len(case["tools"]),
        "fact_score": sum(facts) / len(facts),
        "steps": len(result["tools_used"]),
        "words": len(answer.split()),
    }
    if "verdict" in case:
        m = re.search(r"VERDICT:\s*(GO WITH CAUTION|NO-GO|GO)", answer)
        metrics["verdict_correct"] = bool(m) and m.group(1) == case["verdict"]
    metrics["quality"] = round((metrics["completed"] + metrics["format_ok"] + metrics["tool_recall"] + metrics["fact_score"]) / 4, 3)
    return metrics


def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(line)["run_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


_lock = threading.Lock()


def worker(model_key, runs, total, counter):
    for spec in runs:
        case = CASES[spec["case"]]
        result = ct.run_agent(case["question"], persona=case["persona"], model=MODELS[model_key],
                              config_override=spec["override"], variant=spec["variant"], log=False)
        record = {**spec, "persona": case["persona"], "question": case["question"], "model": MODELS[model_key]["model"],
                  "cfg": result["cfg"], "metrics": score(spec["case"], result), "error": result["error"],
                  "final_answer": result["final_answer"], "tools_used": result["tools_used"], "steps": result["steps"],
                  "llm_calls": result["llm_calls"], "format_errors": result["format_errors"],
                  "latency_s": result["latency_s"], "tokens_out": result["tokens_out"], "ignored_params": result["ignored_params"],
                  "native_tool_calls": result["native_tool_calls"], "implicit_finals": result["implicit_finals"],
                  "ungrounded_finals": result["ungrounded_finals"]}
        with _lock:
            with open(RUNS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            counter[0] += 1
            m = record["metrics"]
            print(f"[{counter[0]}/{total}] {spec['run_id']:<48} quality={m['quality']:.2f} "
                  f"facts={m['fact_score']:.2f} fmt={'ok' if m['format_ok'] else 'X'} "
                  f"steps={m['steps']} {'ERROR ' + record['error'][:60] if record['error'] else ''}", flush=True)


def run_all(only=None):
    todo = [r for r in plan() if r["run_id"] not in done_ids(RUNS_FILE) and (not only or r["exp"] == only)]
    by_model = defaultdict(list)
    for r in todo:
        by_model[r["model_key"]].append(r)
    print(f"{len(todo)} runs to do: " + ", ".join(f"{m}={len(v)}" for m, v in by_model.items()), flush=True)
    counter = [0]
    threads = [threading.Thread(target=worker, args=(m, runs, len(todo), counter)) for m, runs in by_model.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("Done. Results in", RUNS_FILE)


JUDGE_PROMPT = """You are grading a hiking-planner assistant. The assistant was asked to play this persona:
{persona}

Hiker request: {question}
Assistant's final answer:
<<<
{answer}
>>>

Score two things from 1 (poor) to 5 (excellent):
- persona_fit: does the tone, vocabulary and style match the persona description?
- helpfulness: is the answer useful, specific and appropriate for the request?
Reply with JSON only: {{"persona_fit": <1-5>, "helpfulness": <1-5>, "comment": "<max 15 words>"}}"""


def judge_all():
    judged = done_ids(JUDGE_FILE)
    runs = [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [r for r in runs if r["run_id"] not in judged and not r["error"]]
    print(f"{len(todo)} runs to judge with {ct.GEMINI_MODEL}", flush=True)
    for i, r in enumerate(todo, 1):
        p = ct.PERSONAS[r["persona"]]
        prompt = JUDGE_PROMPT.format(persona=p["prompt"], question=r["question"], answer=r["final_answer"])
        try:
            text, _ = ct.call_llm("You are a strict, fair evaluator.", [{"role": "user", "content": prompt}],
                                  {"provider": "gemini", "model": ct.GEMINI_MODEL, "temperature": 0.0, "max_tokens": 200})
            scores = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {r['run_id']} judge failed: {exc}", flush=True)
            continue
        with open(JUDGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": r["run_id"], **scores}, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {r['run_id']:<48} persona_fit={scores.get('persona_fit')} helpfulness={scores.get('helpfulness')}", flush=True)


def redo_affected():
    """Set aside runs affected by the parser fix (format errors or crashes) so they are re-run."""
    runs = [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    redo = {r["run_id"] for r in runs if r["format_errors"] > 0 or r["error"] or not r["tools_used"]}
    for path in (RUNS_FILE, JUDGE_FILE):
        if not path.exists():
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        keep = [line for line in lines if json.loads(line)["run_id"] not in redo]
        dropped = [line for line in lines if json.loads(line)["run_id"] in redo]
        with open(path.with_name(path.stem + "_discarded_v1.jsonl"), "a", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in dropped)
        path.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    print(f"Set aside {len(redo)} runs for re-running.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--redo", action="store_true", help="set aside runs with format errors/crashes, then re-run them")
    args = ap.parse_args()
    if args.dry_run:
        runs = plan()
        counts = defaultdict(int)
        for r in runs:
            counts[(r["exp"], r["model_key"])] += 1
        for k, v in sorted(counts.items()):
            print(k, v)
        print("total", len(runs), "| already done", len(done_ids(RUNS_FILE)))
    elif args.judge:
        judge_all()
    elif args.redo:
        redo_affected()
        run_all(args.only)
    else:
        run_all(args.only)
