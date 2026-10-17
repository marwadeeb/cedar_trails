"""Build cedar_trails.ipynb from the `# %%` sections of cedar_trails.py plus write-up and analysis cells."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
src = (ROOT / "cedar_trails.py").read_text(encoding="utf-8")
sec = {m.group(1): m.group(2).strip("\n") for m in re.finditer(r"^# %% (\w+)\n(.*?)(?=^# %% |\Z)", src, re.S | re.M)}
sec["ui"] = sec["ui"].split('\n\nif __name__ == "__main__":')[0]
reflection = (ROOT / "REFLECTION.md").read_text(encoding="utf-8") if (ROOT / "REFLECTION.md").exists() else "_Reflection pending._"

cells = []
md = lambda t: cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
code = lambda t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md("""
# Cedar Trails: a ReAct hiking planner with configurable personas
**EECE 503P/798S Agentic Systems, Assignment: Agent Personas & Thinking Mechanisms**

Cedar Trails helps hikers plan sections of the **Lebanon Mountain Trail (LMT)**. It is a **ReAct agent built by hand in LangGraph**:
explicit `reason`, `act` and `finish` nodes with conditional edges, text-based `Thought / Action / Observation` steps, our own action
parser, and no prebuilt agent executor. Three personas share the loop but differ in **task, prompt traits and LLM configuration**.

| Section | Content |
|---|---|
| 1 | Use case: goal, audience, tools, constraints |
| 2-3 | Setup, data and tools (MCP-style schemas) |
| 4 | Personas: prompt traits + matching LLM configuration |
| 5 | Dedicated instructions block (with worked example) |
| 6-8 | LLM adapter, action parser, LangGraph ReAct loop |
| 9 | Live demo of the three personas |
| 10 | Experiments: personas x models, persona x config, prompt engineering, parameter sweeps |
| 11 | Reflection |
| 12 | Gradio app |
""")

md("""
## 1. Use case

| | |
|---|---|
| **Agent goal** | Help people plan safe, enjoyable hikes on the Lebanon Mountain Trail: pick sections, estimate time, judge safety, and pack correctly. |
| **Target audience** | Local and diaspora hikers, from beginners planning a first day hike to fit groups attempting multi-day sections; also hiking clubs and guesthouse hosts. |
| **Tools** | `search_trail_sections`, `get_section_details`, `estimate_hike_time` (Naismith's rule), `get_season_conditions`, `check_safety_alerts`, `build_packing_list` |
| **Personas / tasks** | **Rami**, Friendly Trail Buddy: route inspiration. **Hala**, Strict Safety Ranger: go / no-go assessment. **Nour**, Gear Quartermaster: packing list. |
| **Constraints** | Answers must be grounded in tool observations (no invented sections or numbers); at most 5 tool calls per question; safety verdicts follow a fixed rule (danger alert, or high snow risk above 1500 m, means NO-GO). |
| **Limitations** | The trail data is a simplified, illustrative dataset (12 sections, typical seasonal conditions, simulated alerts), not live weather or official closures; time estimates use a rule of thumb; the agent cannot book guesthouses or guides. Real hikers must check lebanontrail.org and official advisories. |
""")

md("""
## 2. Setup
Keys go in `.env` (never committed): `GROQ_API_KEY` (free, used for Qwen3.8-27B and GPT-OSS) and `GEMINI_API_KEY` (free tier).
""")
code("%pip install -q -r requirements.txt")
code(sec["imports"])
code(sec["config"])
code('print(DATA["_note"])\nprint(len(DATA["sections"]), "sections:", ", ".join(s["id"] + " " + s["name"] for s in DATA["sections"][:4]), "...")')

md("""
## 3. Tools
Six deterministic Python tools. Each is documented MCP-style (`name`, `description`, `inputSchema`); the same descriptions are
rendered into the prompt (tool descriptions embedded in the instructions, as the assignment requires).
""")
code(sec["tools"])
code('print(describe_tools())\nprint(json.dumps(estimate_hike_time(11, 400, "beginner"), indent=1))\nprint(execute_tool("get_section_details", {"section_id": "S99"}))  # errors become observations')

md("""
## 4. Personas: prompt traits + matching LLM configuration
Each persona is defined by **prompt wording** (tone, vocabulary, response style, values) **and** an **LLM configuration chosen to support it**:

| Persona | Task | Prompt traits | Config | Why this config fits |
|---|---|---|---|---|
| **Rami**, Friendly Trail Buddy | Route inspiration | warm, vivid, conversational, encouraging | temp 0.8, top_p 0.95, top_k 64, presence 0.4, 700 tokens | variety and fresh wording for inspiration |
| **Hala**, Strict Safety Ranger | Go / no-go | firm, precise, technical, verdict first | temp 0.1, top_p 0.5, top_k 10, no penalties, 450 tokens | the same facts should always give the same verdict; repetition of exact terms is good |
| **Nour**, Gear Quartermaster | Packing list | practical, concrete quantities, dry humour | temp 0.3, top_p 0.8, top_k 40, frequency 0.5, 600 tokens | consistent lists; frequency penalty discourages duplicate items |
""")
code(sec["personas"])

md("""
## 5. Dedicated instructions block
Built following the class guidelines:
* **One purpose per instruction**: every numbered rule covers a single concern (format, grounding, limit...).
* **Explicit inputs and outputs**: rule 1 states the input; each persona has an exact output format for the final answer.
* **A worked example** inside the instructions (a full Thought / Action / Observation trace).
* **No conflict with the persona**: tone lives in the persona; the instructions only govern process and format, and each output format leaves room for the persona's voice (Rami's tip, Nour's note).

The block is parameterised for the experiments: chain-of-thought on/off and few-shot on/off.
""")
code(sec["instructions"])
code('print(build_instructions("hala"))')

md("""
## 6. LLM adapter
One function, three backends. It records which parameters a backend **ignored**:

| Parameter | Gemini 3.5 flash-lite | Groq (Qwen3.8-27B, GPT-OSS) |
|---|---|---|
| temperature, top_p, max_tokens, seed, stop | yes | yes |
| top_k | yes | no (not in the OpenAI-style API) |
| presence / frequency penalty | **rejected** ("Penalty is not enabled for this model") | yes |

Gemini's hidden thinking tokens also count against `max_output_tokens`, and GPT-OSS has a hidden reasoning channel (set to `low`).
""")
code(sec["llm"])

md("""
## 7. Action detection and parsing
The crucial step: detect when the model proposes an **Action**, parse its arguments (JSON or Python keyword style), recognise a
**Final Answer**, and turn anything else into a format error that is fed back to the model as an observation. Robustness features
added after the experiments revealed real failure modes:
* invented observations are cut off (the model must never write `Observation:` itself);
* a reply that starts the persona's output format without `Final Answer:` is accepted as an **implicit final** (GPT-OSS did this);
* a GPT-OSS call on its **native tool channel** is converted back to a text Action;
* hallucinated tool names (e.g. `repo_browser.open_file`) are parsed and answered with "unknown tool".
""")
code(sec["parser"])
code("""
for text in ['Thought: search first\\nAction: search_trail_sections({"region": "north"})',
             'Action: `estimate_hike_time(distance_km=9, elevation_gain_m=350, fitness="fit")`',
             'Action: check_safety_alerts({"section_id": "S03"})\\nObservation: {"made": "up"}',
             '**Top pick:** Falougha to Ain Zhalta (S08)',
             'I think the trail is nice.']:
    s = parse_step(text, FINAL_MARKERS["rami"])
    print(f"{s['kind']:8} tool={s['tool']} args={s['args']} implicit={s['implicit']} error={s['error']}")
""")

md("""
## 8. The ReAct loop in LangGraph
```
START -> reason --(Action parsed)--> act --(Observation)--> reason
           |  \\--(format / grounding error: feedback observation)--> reason
           \\--(Final Answer or step limit)--> finish -> END
```
* **reason** = Thought (+ Action or Final Answer): one LLM call, then action detection.
* **act** = execute the tool and append `Observation: ...` to the scratchpad.
* **finish** = return the final answer (or a safe fallback at the step limit).
* A **grounding guard** rejects a final answer that arrives before any tool call.
""")
code(sec["graph"])
code("""
try:
    from IPython.display import Image, display
    display(Image(GRAPH.get_graph().draw_mermaid_png()))
except Exception as exc:  # needs internet (mermaid.ink); fall back to text
    print(GRAPH.get_graph().draw_mermaid())
""")
code(sec["run"])

md("## 9. Live demo: one task per persona\nUses `DEFAULT_MODEL`: Qwen3.8-27B via Groq, unless `AGENT_MODEL` is set in `.env`.")
code("""
for persona, question in [("rami", "An easy day hike in the Shouf in October for two beginners?"),
                          ("hala", "We plan Sir ed-Danniyeh to Ehden in January with a fit group. Go or no-go?"),
                          ("nour", "What should I pack for 3 days on the trail in the north in August?")]:
    r = run_agent(question, persona=persona)
    print("=" * 90, f"\\n{PERSONAS[persona]['name']} ({PERSONAS[persona]['title']}) | model: {r['cfg']['model']} | tools: {r['tools_used']} | "
          f"LLM calls: {r['llm_calls']} | format errors: {r['format_errors']}\\n")
    print(format_trace(r))
""")

md("""
## 10. Experiments
All runs are produced by `experiments.py` (resumable, one thread per model) and stored in `results/runs.jsonl`; an LLM judge
(Gemini, temperature 0) scores **persona fit** and **helpfulness** (1-5) in `results/judge.jsonl`.

**Automatic metrics per run:** `completed`, `format_ok` (final answer follows the persona's output format), `tool_recall`
(required tools actually used), `fact_score` (checkable facts from observations present, e.g. 4.5 h, S08, NO-GO), `quality`
(mean of those four), Hala's `verdict_correct`, plus format errors, ungrounded attempts, steps, latency.

| Exp | Question | Design | Model |
|---|---|---|---|
| E1 | Which model works best as the ReAct engine, per persona? | 3 personas x 2 tasks x 4 models | Gemini 3.5 flash-lite, Qwen3.8-27B, GPT-OSS-120B, GPT-OSS-20B |
| E2 | Does each persona's own config suit it best? | 6 tasks x 3 configs (each persona's) | GPT-OSS-120B |
| E3 | Prompt engineering: CoT, few-shot, placement | 2 x 2 x 2 variants x 3 tasks | GPT-OSS-20B |
| E4 | Single-parameter sweeps (3 repeats each) | temperature, top_p, penalties, max_tokens; top_k | Qwen3.8-27B; Gemini for top_k |
| E5 | Instructions ablation; last-step budget warning | full vs minimal instructions x 6 tasks; warning on/off x 2 tasks x 2 reps | Qwen3.8-27B (+ GPT-OSS-20B for E5b) |
""")
code("""
import json
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

runs = [json.loads(l) for l in open("results/runs.jsonl", encoding="utf-8") if l.strip()]
judge = {j["run_id"]: j for j in (json.loads(l) for l in open("results/judge.jsonl", encoding="utf-8") if l.strip())}
df = pd.DataFrame([{**{k: r[k] for k in ("run_id", "exp", "model_key", "case", "persona", "label", "rep", "final_answer")},
                    **r["metrics"], "format_errors": r["format_errors"], "ungrounded": r.get("ungrounded_finals", 0),
                    "native_calls": r.get("native_tool_calls", 0), "implicit": r.get("implicit_finals", 0),
                    "latency_s": r["latency_s"], "error": bool(r["error"]),
                    "persona_fit": judge.get(r["run_id"], {}).get("persona_fit"),
                    "helpfulness": judge.get(r["run_id"], {}).get("helpfulness")} for r in runs])
AGG = dict(quality=("quality", "mean"), facts=("fact_score", "mean"), format_ok=("format_ok", "mean"),
           tool_recall=("tool_recall", "mean"), format_errors=("format_errors", "mean"), ungrounded=("ungrounded", "mean"),
           persona_fit=("persona_fit", "mean"), helpfulness=("helpfulness", "mean"), words=("words", "mean"), n=("quality", "size"))
print(len(df), "runs |", df.error.sum(), "provider errors |", df.persona_fit.notna().sum(), "judged")
""")

md("### E1: persona x model")
code("""
e1 = df[df.exp == "E1"]
display(e1.groupby("model_key").agg(**AGG).round(2).sort_values("quality", ascending=False))
pivot = e1.pivot_table(index="persona", columns="model_key", values="quality", aggfunc="mean")
ax = pivot.plot.bar(figsize=(9, 3.5), rot=0, title="E1: answer quality by persona and model")
ax.set_ylim(0, 1.05); ax.set_ylabel("quality (0-1)"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()
print("Hala verdict accuracy by model:")
print(e1[e1.persona == "hala"].groupby("model_key")["verdict_correct"].mean())
""")

md("### E2: does each persona's own configuration suit it best?")
code("""
e2 = df[df.exp == "E2"]
heat = e2.pivot_table(index="persona", columns="label", values="quality", aggfunc="mean")
display(e2.groupby(["persona", "label"]).agg(**AGG).round(2))
fig, ax = plt.subplots(figsize=(5.5, 3))
im = ax.imshow(heat.values, cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(len(heat.columns)), heat.columns); ax.set_yticks(range(len(heat.index)), heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center")
ax.set_title("E2: quality, persona (rows) x config used (cols)"); plt.colorbar(im); plt.tight_layout(); plt.show()
""")

md("### E3: prompt engineering (CoT x few-shot x system vs in-message instructions)")
code("""
e3 = df[df.exp == "E3"].copy()
e3["cot"] = e3.label.str.startswith("cot"); e3["few_shot"] = e3.label.str.contains("_few_"); e3["in_system"] = e3.label.str.endswith("system")
display(e3.groupby("label").agg(**AGG).round(2).sort_values("quality", ascending=False))
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax, factor, names in zip(axes, ["cot", "few_shot", "in_system"], [("no CoT", "CoT"), ("zero-shot", "few-shot"), ("in-message", "system")]):
    vals = e3.groupby(factor)[["quality", "tool_recall"]].mean()
    vals.index = [names[int(i)] for i in vals.index]
    vals.plot.bar(ax=ax, rot=0, legend=ax is axes[0], ylim=(0, 1), title=factor)
plt.suptitle("E3: main effect of each prompt choice"); plt.tight_layout(); plt.show()
""")

md("### E4: parameter sweeps (Rami's route task, 3 repeats each)")
code("""
def diversity(answers):
    \"\"\"Mean pairwise Jaccard distance between word sets: 0 = identical answers, 1 = nothing shared.\"\"\"
    sets = [set(a.lower().split()) for a in answers]
    pairs = list(combinations(sets, 2))
    return round(sum(1 - len(a & b) / len(a | b) for a, b in pairs) / len(pairs), 2) if pairs else None

e4 = df[df.exp == "E4"]
table = e4.groupby(["model_key", "label"]).agg(**AGG).round(2)
table["diversity"] = e4.groupby(["model_key", "label"])["final_answer"].apply(lambda s: diversity(list(s)))
display(table)
temps = table.loc["qwen3.8-27b"].filter(like="temp_", axis=0)
temps.index = [float(i.split("_")[1]) for i in temps.index]
ax = temps[["quality", "diversity"]].sort_index().plot(marker="o", figsize=(6, 3), title="E4: temperature vs quality and diversity (Qwen3.8-27B)")
ax.set_xlabel("temperature"); ax.set_ylim(0, 1.05); plt.tight_layout(); plt.show()
""")
code("""
# What does each setting actually produce? One answer per temperature.
for label in ["temp_0.0", "temp_0.8", "temp_1.6"]:
    print(f"--- {label} ---\\n{e4[(e4.label == label)].iloc[0].final_answer[:500]}\\n")
""")

md("""
### E5: does the instructions block matter, and does a last-step warning fix step-limit failures?
E1-E4 showed that the most common failure was **running out of steps** (repeated searches or repeated tool calls), so we tested a
one-line **budget warning** on the last step (E5b, on the tasks that failed in E1). E5a removes the dedicated instructions block
(keeping only the tool list and the action syntax). Two Qwen runs of E5b were skipped because Groq's free daily token quota
(200k tokens/day) ran out; that cell has n = 2 instead of 4.
""")
code("""
e5 = df[df.exp == "E5"]
display(e5[e5.label.str.startswith("instr")].groupby("label").agg(**AGG).round(2))
display(e5[e5.label.str.startswith("instr")].pivot_table(index="case", columns="label", values="quality").round(2))
display(e5[e5.label.str.startswith("budget")].groupby(["model_key", "label"]).agg(**AGG).round(2))
""")
code("""
# Same question, with and without the instructions block (Hala, H1)
for label in ["instr_full", "instr_minimal"]:
    print(f"--- {label} ---\\n{e5[(e5.label == label) & (e5.case == 'H1')].iloc[0].final_answer[:600]}\\n")
""")

md("## 11. Reflection\n" + reflection)

md("## 12. Gradio app\nPick a persona and a model; the right panel shows the full ReAct trace for every reply.")
code(sec["ui"])
code("demo = build_ui()\ndemo.launch()")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
(ROOT / "cedar_trails.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells")
