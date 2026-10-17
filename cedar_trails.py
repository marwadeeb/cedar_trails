"""Cedar Trails: a ReAct hiking-planner agent for the Lebanon Mountain Trail, built on LangGraph.

The ReAct loop is implemented by hand as LangGraph nodes and conditional edges
(no prebuilt agent executor). The `# %%` sections below are also the cells of
cedar_trails.ipynb.
"""

# %% imports
import ast
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

# %% config
load_dotenv(override=True)

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # inside a notebook
    BASE_DIR = Path.cwd()

DATA = json.loads((BASE_DIR / "data" / "trail_data.json").read_text(encoding="utf-8"))
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MAX_STEPS = 6  # maximum reason->act cycles per question

# %% tools
SECTIONS = {s["id"]: s for s in DATA["sections"]}
SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def _month(value):
    """Accept 7, "7", "July" or "jul" and return 1-12."""
    if isinstance(value, (int, float)) or str(value).strip().isdigit():
        month = int(value)
    else:
        text = str(value).strip().lower()[:3]
        month = next((i + 1 for i, m in enumerate(MONTHS) if m.startswith(text)), 0)
    if not 1 <= month <= 12:
        raise ValueError(f"unknown month: {value!r}")
    return month


def _section(section_id):
    section = SECTIONS.get(str(section_id).strip().upper())
    if section is None:
        raise ValueError(f"unknown section_id {section_id!r}; valid ids are S01 to S12")
    return section


def search_trail_sections(region="any", difficulty="any", max_distance_km=None, month=None):
    """Find trail sections matching optional filters."""
    matches = []
    for s in DATA["sections"]:
        if region != "any" and s["region"] != region:
            continue
        if difficulty != "any" and s["difficulty"] != difficulty:
            continue
        if max_distance_km is not None and s["distance_km"] > float(max_distance_km):
            continue
        if month is not None and _month(month) not in s["best_months"]:
            continue
        matches.append({k: s[k] for k in ("id", "name", "region", "difficulty", "distance_km", "elevation_gain_m", "highlights")})
    return {"count": len(matches), "sections": matches}


def get_section_details(section_id):
    """Full record for one section."""
    s = _section(section_id)
    return {**s, "best_months": [MONTHS[m - 1].title() for m in s["best_months"]]}


def estimate_hike_time(distance_km, elevation_gain_m, fitness="average"):
    """Naismith's rule: 5 km/h on the flat + 1 hour per 600 m of ascent, scaled by fitness."""
    factor = {"beginner": 1.3, "average": 1.0, "fit": 0.85}.get(str(fitness).lower())
    if factor is None:
        raise ValueError("fitness must be beginner, average or fit")
    moving = (float(distance_km) / 5 + float(elevation_gain_m) / 600) * factor
    return {
        "moving_hours": round(moving, 1),
        "total_hours_with_breaks": round(moving * 1.2, 1),
        "fitness_factor": factor,
        "method": "Naismith's rule (5 km/h + 1 h per 600 m ascent), +20% for breaks",
    }


def get_season_conditions(region, month):
    """Typical conditions for a region in a given month."""
    if region not in DATA["seasons"]:
        raise ValueError(f"region must be one of {list(DATA['seasons'])}")
    m = _month(month)
    season = SEASON_OF_MONTH[m]
    c = DATA["seasons"][region][season]
    return {"region": region, "month": MONTHS[m - 1].title(), "season": season,
            "temp_range_c": c["temp_c"], "snow_risk": c["snow_risk"], "daylight_hours": c["daylight_h"]}


def check_safety_alerts(section_id):
    """Simulated safety alerts for a section."""
    s = _section(section_id)
    alerts = DATA["alerts"].get(s["id"], [])
    worst = max((a["level"] for a in alerts), key=["info", "warning", "danger"].index, default="none")
    return {"section_id": s["id"], "highest_level": worst, "alerts": alerts}


def build_packing_list(days, month, overnight=False):
    """Packing list for a hike, adapted to season and duration."""
    days, season = max(1, int(days)), SEASON_OF_MONTH[_month(month)]
    essentials = ["map + offline GPS track", "first-aid kit", "headlamp", "sun hat + sunscreen",
                  "rain shell", "trail snacks", "water bottles or bladder", "phone + power bank", "whistle"]
    extras = {
        "winter": ["insulated jacket", "gloves + beanie", "microspikes or crampons", "gaiters", "thermos"],
        "spring": ["warm mid-layer", "light gloves", "gaiters for muddy trails"],
        "summer": ["electrolyte tablets", "lightweight long sleeves", "extra water capacity"],
        "autumn": ["warm mid-layer", "light gloves", "extra daylight margin (headlamp batteries)"],
    }[season]
    overnight_items = ["guesthouse booking confirmation", "sleep layer", "toiletries", "spare socks"] if overnight or days > 1 else []
    water = 3.5 if season == "summer" else 2.5
    weight = 6 + 1.2 * (days - 1) + (2 if season == "winter" else 0)
    return {"days": days, "season": season, "essentials": essentials, "season_extras": extras,
            "overnight": overnight_items, "water_liters_per_day": water, "estimated_pack_kg": round(weight, 1)}


# MCP-style tool definitions: name, description, inputSchema (+ the callable).
TOOLS = {
    "search_trail_sections": {
        "fn": search_trail_sections,
        "description": "Find LMT sections by region, difficulty, maximum distance and/or a month in their best season.",
        "inputSchema": {"type": "object", "properties": {
            "region": {"type": "string", "enum": ["any", "north", "mount_lebanon", "shouf", "south"]},
            "difficulty": {"type": "string", "enum": ["any", "easy", "moderate", "hard"]},
            "max_distance_km": {"type": "number"},
            "month": {"type": "string", "description": "Month name or number"}}},
    },
    "get_section_details": {
        "fn": get_section_details,
        "description": "Full stats for one section: distance, elevation, best months, water points, guesthouse.",
        "inputSchema": {"type": "object", "properties": {"section_id": {"type": "string", "description": "S01 to S12"}},
                        "required": ["section_id"]},
    },
    "estimate_hike_time": {
        "fn": estimate_hike_time,
        "description": "Estimate hiking hours from distance and elevation gain (Naismith's rule) for a fitness level.",
        "inputSchema": {"type": "object", "properties": {
            "distance_km": {"type": "number"}, "elevation_gain_m": {"type": "number"},
            "fitness": {"type": "string", "enum": ["beginner", "average", "fit"]}},
            "required": ["distance_km", "elevation_gain_m"]},
    },
    "get_season_conditions": {
        "fn": get_season_conditions,
        "description": "Typical temperature range, snow risk and daylight for a region in a month.",
        "inputSchema": {"type": "object", "properties": {
            "region": {"type": "string", "enum": ["north", "mount_lebanon", "shouf", "south"]},
            "month": {"type": "string"}}, "required": ["region", "month"]},
    },
    "check_safety_alerts": {
        "fn": check_safety_alerts,
        "description": "Current (simulated) safety alerts and closures for a section.",
        "inputSchema": {"type": "object", "properties": {"section_id": {"type": "string"}}, "required": ["section_id"]},
    },
    "build_packing_list": {
        "fn": build_packing_list,
        "description": "Season-aware packing list with water per day and estimated pack weight.",
        "inputSchema": {"type": "object", "properties": {
            "days": {"type": "integer"}, "month": {"type": "string"}, "overnight": {"type": "boolean"}},
            "required": ["days", "month"]},
    },
}


def describe_tools():
    """Tool descriptions embedded in the prompt."""
    lines = []
    for name, t in TOOLS.items():
        props = t["inputSchema"]["properties"]
        required = set(t["inputSchema"].get("required", []))
        args = ", ".join(f"{p}{'' if p in required else '?'}: {spec['type']}"
                         + (f" ({'|'.join(spec['enum'])})" if "enum" in spec else "") for p, spec in props.items())
        lines.append(f"- {name}({args}): {t['description']}")
    return "\n".join(lines)


def execute_tool(name, args):
    """Run a tool and return its observation (errors are returned, not raised)."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}. Available: {', '.join(TOOLS)}"}
    allowed = tool["inputSchema"]["properties"]
    try:
        return tool["fn"](**{k: v for k, v in args.items() if k in allowed})
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}

# %% personas
# Each persona = prompt traits (tone, vocabulary, response style, values) + a matching LLM configuration.
PERSONAS = {
    "rami": {
        "name": "Rami", "title": "Friendly Trail Buddy", "task": "Route inspiration: suggest sections that fit the hiker",
        "prompt": (
            "You are Rami, the friendly trail buddy of Cedar Trails, a hiking planner for the Lebanon Mountain Trail. "
            "Tone: warm, upbeat and encouraging, like a friend who has walked every section. "
            "Vocabulary: vivid but simple; you paint the scenery (cedars, valleys, villages) in a few words. "
            "Response style: conversational, short paragraphs, one practical tip at the end. "
            "Values: you want people to fall in love with the trail, and you never oversell a route beyond their fitness."
        ),
        "config": {"temperature": 0.8, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.4,
                   "frequency_penalty": 0.0, "max_tokens": 700},
        "why": {
            "temperature 0.8": "more varied, lively descriptions for inspiration",
            "top_p 0.95 / top_k 64": "a wide vocabulary pool so suggestions don't sound templated",
            "presence_penalty 0.4": "nudges toward new words and ideas instead of repeating 'beautiful'",
            "max_tokens 700": "room for a short story-like pitch plus stats",
        },
    },
    "hala": {
        "name": "Hala", "title": "Strict Safety Ranger", "task": "Go / no-go safety assessment for a planned hike",
        "prompt": (
            "You are Hala, the strict safety ranger of Cedar Trails. "
            "Tone: calm, firm and direct; you do not soften bad news. "
            "Vocabulary: precise and technical (elevation gain, snow risk, daylight margin), no exclamation marks. "
            "Response style: structured verdict first, then numbered reasons, then required actions. "
            "Values: safety over enthusiasm; when information is missing or alarming, you lean toward NO-GO."
        ),
        "config": {"temperature": 0.1, "top_p": 0.5, "top_k": 10, "presence_penalty": 0.0,
                   "frequency_penalty": 0.0, "max_tokens": 450},
        "why": {
            "temperature 0.1": "near-deterministic: the same inputs should always give the same verdict",
            "top_p 0.5 / top_k 10": "only high-confidence tokens; no creative wording in a safety call",
            "penalties 0": "repeating exact terms (NO-GO, snow risk) is desirable for clarity",
            "max_tokens 450": "forces a concise verdict",
        },
    },
    "nour": {
        "name": "Nour", "title": "Gear Quartermaster", "task": "Packing list and logistics for a trip",
        "prompt": (
            "You are Nour, the gear quartermaster of Cedar Trails. "
            "Tone: practical and efficient, with a dry sense of humour. "
            "Vocabulary: concrete item names and quantities (liters, kilograms), no vague advice. "
            "Response style: grouped checklists, nothing longer than one line per item. "
            "Values: carry what you need and nothing more; weight is the enemy."
        ),
        "config": {"temperature": 0.3, "top_p": 0.8, "top_k": 40, "presence_penalty": 0.0,
                   "frequency_penalty": 0.5, "max_tokens": 600},
        "why": {
            "temperature 0.3": "mostly consistent lists with a little room for a witty line",
            "top_p 0.8 / top_k 40": "moderate diversity",
            "frequency_penalty 0.5": "discourages listing the same item twice",
            "max_tokens 600": "enough for a full checklist, not an essay",
        },
    },
}

# %% instructions
# Dedicated instructions block: one purpose per rule, explicit input/output, a worked example.
PROTOCOL_RULES = [
    "INPUT: you receive a hiker's request and, sometimes, earlier conversation turns.",
    "TOOLS: you may only use the tools listed below.",
    "ACTION FORMAT: to use a tool, write one line exactly like `Action: tool_name({\"arg\": value})` with JSON arguments, then stop.",
    "ONE STEP: write at most one Action per reply.",
    "OBSERVATIONS: never write `Observation:` yourself; the system adds it after running the tool.",
    "GROUNDING: every number you state (km, metres, hours, degrees, litres) must come from an Observation.",
    "LIMIT: use at most 5 Actions, then answer with what you have.",
    "FINAL FORMAT: when you have enough information, write `Final Answer:` followed by the answer in your task's output format.",
]
COT_RULE = "THINKING: before every Action or Final Answer, write one line starting with `Thought:` that reasons step by step about what you know and what you need next."
NO_COT_RULE = "NO THINKING TEXT: do not write any reasoning. Each reply is only an `Action:` line or a `Final Answer:`."

TASK_INSTRUCTIONS = {
    "rami": {
        "rules": [
            "GOAL: recommend the best-fitting section(s) for the hiker's region, month, distance and fitness.",
            "REQUIRED TOOLS: search with search_trail_sections, then estimate_hike_time for your top pick.",
            "HONESTY: if no section fits, say so and suggest the closest alternative.",
        ],
        "output": "**Top pick:** <name> (<id>) - one vivid sentence\n**Stats:** <km> km, +<m> m, about <hours> h for a <fitness> hiker\n**Also consider:** <name> (<id>) - one sentence\n**Rami's tip:** <one practical tip>",
    },
    "hala": {
        "rules": [
            "GOAL: decide GO, GO WITH CAUTION or NO-GO for the planned section, month and group.",
            "REQUIRED TOOLS: check_safety_alerts and get_season_conditions for the section's region; get_section_details when you need the region or stats.",
            "DECISION RULE: any `danger` alert, or high snow risk on a section above 1500 m, means NO-GO.",
        ],
        "output": "VERDICT: <GO | GO WITH CAUTION | NO-GO>\nREASONS:\n1. <reason with a number from an observation>\n2. <reason>\nREQUIRED ACTIONS:\n- <action>",
    },
    "nour": {
        "rules": [
            "GOAL: produce a packing list for the trip the hiker describes.",
            "REQUIRED TOOLS: build_packing_list; add get_season_conditions when a region is known.",
            "BREVITY: one line per item, no duplicates.",
        ],
        "output": "### Essentials\n- [ ] <item>\n### For <season>\n- [ ] <item>\n### Overnight (if any)\n- [ ] <item>\n**Water:** <L> L/day | **Pack weight:** ~<kg> kg\n**Nour's note:** <one dry, useful line>",
    },
}

# One worked example (the same trace is shown with or without Thought lines).
WORKED_EXAMPLE_STEPS = [
    ("The hiker wants an easy section in Mount Lebanon in July and is a beginner. I should search first.",
     'Action: search_trail_sections({"region": "mount_lebanon", "difficulty": "easy", "month": "July"})',
     '{"count": 1, "sections": [{"id": "S07", "name": "Baskinta to Zaarour", "distance_km": 9.0, "elevation_gain_m": 350}]}'),
    ("S07 fits. I need a time estimate for a beginner.",
     'Action: estimate_hike_time({"distance_km": 9.0, "elevation_gain_m": 350, "fitness": "beginner"})',
     '{"moving_hours": 3.1, "total_hours_with_breaks": 3.7}'),
]
WORKED_EXAMPLE_FINAL = ("I have the section and the time, so I can answer.",
                        "Final Answer: Baskinta to Zaarour (S07): 9 km and +350 m, about 3.7 h with breaks for a beginner.")


def worked_example(cot=True):
    lines = ["Hiker request: An easy hike in Mount Lebanon in July for a beginner?"]
    for thought, action, observation in WORKED_EXAMPLE_STEPS:
        lines += ([f"Thought: {thought}"] if cot else []) + [action, f"Observation: {observation}"]
    thought, final = WORKED_EXAMPLE_FINAL
    lines += ([f"Thought: {thought}"] if cot else []) + [final]
    return "\n".join(lines)


MINIMAL_RULES = [  # ablation baseline: only what the loop needs to function
    "To use a tool, write `Action: tool_name({\"arg\": value})` and stop. When done, write `Final Answer:` and your answer.",
]


def build_instructions(persona_id, cot=True, few_shot=True, minimal=False):
    """Assemble the instructions block for a persona and prompt variant."""
    if minimal:
        return "## TOOLS\n" + describe_tools() + "\n\n" + MINIMAL_RULES[0]
    task = TASK_INSTRUCTIONS[persona_id]
    rules = PROTOCOL_RULES + [COT_RULE if cot else NO_COT_RULE] + task["rules"]
    parts = [
        "## INSTRUCTIONS",
        "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1)),
        "## AVAILABLE TOOLS\n" + describe_tools(),
        "## OUTPUT FORMAT FOR THE FINAL ANSWER\n" + task["output"],
    ]
    if few_shot:
        parts.append("## WORKED EXAMPLE (format only; different task)\n" + worked_example(cot))
    return "\n\n".join(parts)

# %% llm
_last_gemini_call = 0.0


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def call_llm(system, messages, cfg):
    """One completion. cfg: provider, model, temperature, top_p, top_k, max_tokens, penalties, seed.

    Returns (text, meta). meta lists parameters the backend ignored, latency and output tokens.
    """
    provider = cfg.get("provider", "gemini")
    stop = ["\nObservation:", "Observation:"]
    t0 = time.time()
    if provider == "gemini":
        from google import genai
        from google.genai import errors, types

        global _last_gemini_call
        client = genai.Client(api_key=GEMINI_API_KEY)
        params = {k: cfg[k] for k in ("temperature", "top_p", "top_k", "seed") if cfg.get(k) is not None}
        # Gemini flash-lite rejects penalties ("Penalty is not enabled for this model").
        ignored = [k for k in ("presence_penalty", "frequency_penalty") if cfg.get(k)]
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=cfg.get("max_tokens"),
                                             stop_sequences=stop, **params)
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
                    for m in messages]
        empty_retries = 0
        for attempt in range(8):
            wait = _last_gemini_call + 4.5 - time.time()  # free tier: 15 requests/min
            if wait > 0:
                time.sleep(wait)
            _last_gemini_call = time.time()
            try:
                r = client.models.generate_content(model=cfg.get("model", GEMINI_MODEL), contents=contents, config=config)
            except (errors.ClientError, errors.ServerError) as exc:
                # 429 = rate limit, 503 = "high demand": back off and retry.
                if getattr(exc, "code", None) not in (429, 500, 503) or attempt == 7:
                    raise
                hint = re.search(r"retry in ([\d.]+)s", str(exc))
                time.sleep(float(hint.group(1)) + 1 if hint else min(60, 5 * 2 ** attempt))
                continue
            if r.text or empty_retries >= 2:
                break
            empty_retries += 1  # empty / MALFORMED_RESPONSE: a provider glitch, not a format error
        text = r.text or ""
        tokens = getattr(r.usage_metadata, "candidates_token_count", None)
    elif provider == "groq":
        # OpenAI-compatible REST API; no top_k parameter exists there.
        ignored = ["top_k"] if cfg.get("top_k") is not None else []
        body = {"model": cfg["model"], "messages": [{"role": "system", "content": system}] + messages, "stop": stop,
                **{k: cfg[k] for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed")
                   if cfg.get(k) is not None}}
        if cfg.get("max_tokens"):
            body["max_completion_tokens"] = cfg["max_tokens"]
        if cfg["model"].startswith("openai/gpt-oss"):
            body["reasoning_effort"] = cfg.get("reasoning_effort", "low")  # hidden reasoning; keep it small
        for attempt in range(6):
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
            if r.status_code == 429 and "tokens per day" in r.text:
                # Daily quota: waiting minutes per call looks like a hang, so fail fast with a clear message.
                raise RuntimeError(f"Groq daily token quota reached for {cfg['model']}; try later or pick another model")
            if attempt == 5 or not (r.status_code in (429, 500, 503) or (r.status_code == 400 and "Parsing failed" in r.text)):
                break
            time.sleep(min(60, float(r.headers.get("retry-after", 0) or 0) + 2 * (attempt + 1)))  # cap long daily-limit waits
        if r.status_code == 400 and '"tool_use_failed"' in r.text:
            # gpt-oss sometimes emits a call on its native tool channel instead of our text format.
            # Groq rejects it (we register no API tools); convert it back to a ReAct Action line.
            raw = r.json()["error"].get("failed_generation") or ""
            try:
                gen = json.loads(raw)
                text = f"Action: {gen.get('name')}({json.dumps(gen.get('arguments', {}))})"
            except json.JSONDecodeError:
                text = raw  # malformed native call: let the parser report a format error
            return text, {"latency_s": round(time.time() - t0, 2), "tokens_out": None, "ignored": ignored, "native_tool_call": True}
        if r.status_code == 400 and "Parsing failed" in r.text:
            # The model produced output Groq could not decode; treat it as an unusable reply.
            return "", {"latency_s": round(time.time() - t0, 2), "tokens_out": None, "ignored": ignored, "provider_parse_error": True}
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
        data = r.json()
        text, tokens = data["choices"][0]["message"].get("content") or "", data.get("usage", {}).get("completion_tokens")
    elif provider == "ollama":
        options = {"temperature": cfg.get("temperature"), "top_p": cfg.get("top_p"), "top_k": cfg.get("top_k"),
                   "presence_penalty": cfg.get("presence_penalty"), "frequency_penalty": cfg.get("frequency_penalty"),
                   "num_predict": cfg.get("max_tokens"), "seed": cfg.get("seed"), "stop": stop}
        body = {"model": cfg["model"], "stream": False, "options": {k: v for k, v in options.items() if v is not None},
                "messages": [{"role": "system", "content": system}] + messages}
        if cfg["model"].startswith("qwen3"):
            body["think"] = False  # qwen3 thinks by default; CoT is controlled by our prompt instead
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300)
        r.raise_for_status()
        data = r.json()
        text, tokens, ignored = data["message"]["content"], data.get("eval_count"), []
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return _strip_think(text), {"latency_s": round(time.time() - t0, 2), "tokens_out": tokens, "ignored": ignored}

# %% parser
ACTION_RE = re.compile(r"Action\s*:\s*`*\s*([A-Za-z_][\w.]*)\s*\(", re.S)
FINAL_RE = re.compile(r"Final\s+Answer\s*:\s*", re.I)


def _parse_args(raw):
    """Parse tool arguments given as JSON or as Python keyword arguments."""
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        return json.loads(raw)
    call = ast.parse(f"f({raw})", mode="eval").body
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}


# The first line of each persona's output format; a reply that starts the answer
# without the `Final Answer:` prefix is still accepted as final (counted as implicit).
FINAL_MARKERS = {"rami": "**Top pick:**", "hala": "VERDICT:", "nour": "### Essentials"}


def parse_step(text, final_marker=None):
    """Detect what the model wants: an Action (tool call), a Final Answer, or neither.

    Returns {"kind": "action"|"final"|"invalid", "thought", "tool", "args", "answer", "error"}.
    """
    text = text.split("Observation:")[0]  # drop any observation the model invented
    action, final = ACTION_RE.search(text), FINAL_RE.search(text)
    thought_end = min(m.start() for m in (action, final) if m) if (action or final) else len(text)
    thought = re.sub(r"^\s*Thought\s*:\s*", "", text[:thought_end].strip(), flags=re.I).strip()
    step = {"kind": "invalid", "thought": thought, "tool": None, "args": None, "answer": None, "error": None, "implicit": False}

    if action and (not final or action.start() < final.start()):
        tail = text[action.end():]
        close = tail.rfind(")")
        step["tool"] = action.group(1)
        if close == -1:
            step["error"] = "Action is missing its closing parenthesis."
            return step
        try:
            step["args"] = _parse_args(tail[:close])
            step["kind"] = "action"
        except (ValueError, SyntaxError) as exc:
            step["error"] = f"Could not parse the Action arguments as JSON: {exc}"
        return step
    if final:
        step["kind"], step["answer"] = "final", text[final.end():].strip()
        return step
    if final_marker and final_marker in text:
        step["kind"], step["implicit"] = "final", True
        step["answer"] = text[text.index(final_marker):].strip()
        step["thought"] = re.sub(r"^\s*Thought\s*:\s*", "", text[:text.index(final_marker)].strip(), flags=re.I).strip()
        return step
    step["error"] = "Reply had neither an `Action:` line nor a `Final Answer:`."
    return step

# %% graph
class AgentState(TypedDict, total=False):
    question: str
    history: list            # earlier conversation turns (short-term memory)
    persona: str
    cfg: dict                # LLM configuration for this run
    variant: dict            # prompt variant: cot, few_shot, placement
    scratchpad: str          # Thought / Action / Observation transcript so far
    steps: list              # structured trace
    pending: Optional[dict]  # parsed action waiting to be executed
    final_answer: Optional[str]
    iterations: int
    format_errors: int
    llm_calls: int
    latency_s: float
    tokens_out: int
    ignored_params: list
    native_tool_calls: int   # times the model used its built-in tool channel instead of text
    implicit_finals: int     # final answers given without the `Final Answer:` prefix
    ungrounded_finals: int   # answers attempted before any tool call (rejected by the grounding guard)


def build_prompt(state):
    """Persona always lives in the system prompt; the instructions go either there or in the user message."""
    v = state["variant"]
    persona = PERSONAS[state["persona"]]["prompt"]
    instructions = build_instructions(state["persona"], cot=v.get("cot", True), few_shot=v.get("few_shot", True),
                                      minimal=v.get("instructions") == "minimal")
    in_system = v.get("placement", "system") == "system"
    system = persona + ("\n\n" + instructions if in_system else "")
    user = ("" if in_system else instructions + "\n\n") + f"Hiker request: {state['question']}\n\n{state['scratchpad']}"
    if v.get("budget_warning") and state["iterations"] == MAX_STEPS - 1:
        # Tested in E5b after E1-E4 showed step-limit failures; it did not help (loops were the cause), so it is off by default.
        user += "\nSYSTEM NOTE: this is your last step. Reply now with the Final Answer using the observations you have."
    return system, state.get("history", []) + [{"role": "user", "content": user.rstrip()}]


def reason_node(state):
    """THOUGHT: ask the model for its next step and detect an Action or a Final Answer."""
    system, messages = build_prompt(state)
    text, meta = call_llm(system, messages, state["cfg"])
    step = parse_step(text, FINAL_MARKERS[state["persona"]])
    update = {
        "iterations": state["iterations"] + 1,
        "llm_calls": state["llm_calls"] + 1,
        "latency_s": state["latency_s"] + meta["latency_s"],
        "tokens_out": state["tokens_out"] + (meta["tokens_out"] or 0),
        "ignored_params": sorted(set(state["ignored_params"]) | set(meta["ignored"])),
        "native_tool_calls": state["native_tool_calls"] + int(meta.get("native_tool_call", False)),
        "pending": None,
    }
    visible = text.split("Observation:")[0].strip()
    if step["kind"] == "action":
        update["pending"] = step
        update["scratchpad"] = state["scratchpad"] + visible + "\n"
    elif step["kind"] == "final" and not any(s["type"] == "action" for s in state["steps"]) \
            and update["iterations"] < MAX_STEPS:
        # Grounding guard: an answer before any tool call cannot be grounded in observations.
        update["ungrounded_finals"] = state["ungrounded_finals"] + 1
        update["scratchpad"] = state["scratchpad"] + visible + (
            "\nObservation: GROUNDING ERROR - you answered without calling any tool, so your numbers "
            "and section names are not verified. Use the tools first, then give the Final Answer.\n")
        update["steps"] = state["steps"] + [{"type": "format_error", "raw": visible, "error": "answered before using any tool"}]
    elif step["kind"] == "final":
        update["final_answer"] = step["answer"]
        update["implicit_finals"] = state["implicit_finals"] + int(step["implicit"])
        update["steps"] = state["steps"] + [{"type": "final", "thought": step["thought"], "answer": step["answer"]}]
    else:
        # Format error: feed the problem back as an observation so the model can self-correct.
        update["format_errors"] = state["format_errors"] + 1
        update["scratchpad"] = state["scratchpad"] + visible + f"\nObservation: FORMAT ERROR - {step['error']} Reply with an Action line or a Final Answer.\n"
        update["steps"] = state["steps"] + [{"type": "format_error", "raw": visible, "error": step["error"]}]
    return update


def act_node(state):
    """ACTION -> OBSERVATION: run the requested tool and append its result to the scratchpad."""
    step = state["pending"]
    observation = execute_tool(step["tool"], step["args"])
    obs_text = json.dumps(observation, ensure_ascii=False)
    return {
        "scratchpad": state["scratchpad"] + f"Observation: {obs_text}\n",
        "steps": state["steps"] + [{"type": "action", "thought": step["thought"], "tool": step["tool"],
                                    "args": step["args"], "observation": observation}],
        "pending": None,
    }


def finish_node(state):
    """ANSWER: make sure there is a final answer even if the step limit was hit."""
    if state.get("final_answer"):
        return {}
    return {"final_answer": "I couldn't complete this plan within my step limit. Please narrow the request (section, month, fitness)."}


def route_after_reason(state):
    """Conditional edge: Action -> act, Final Answer or step limit -> finish, format error -> reason again."""
    if state.get("final_answer"):
        return "finish"
    if state["iterations"] >= MAX_STEPS:
        return "finish"
    return "act" if state.get("pending") else "reason"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)
    g.add_node("finish", finish_node)
    g.add_edge(START, "reason")
    g.add_conditional_edges("reason", route_after_reason, {"act": "act", "finish": "finish", "reason": "reason"})
    g.add_edge("act", "reason")
    g.add_edge("finish", END)
    return g.compile()


GRAPH = build_graph()

# %% run
# Default chosen from experiment E1 (best ReAct reliability). Override with AGENT_MODEL=provider:model in .env,
# e.g. AGENT_MODEL=groq:openai/gpt-oss-20b when Qwen's free daily quota is used up.
_provider, _, _model = os.getenv("AGENT_MODEL", "groq:qwen/qwen3.8-27b").partition(":")
DEFAULT_MODEL = {"provider": _provider, "model": _model}


def run_agent(question, persona="rami", history=None, model=None, config_override=None, variant=None, log=True):
    """Run one ReAct episode and return the final answer, trace and metrics."""
    cfg = {**(model or DEFAULT_MODEL), **PERSONAS[persona]["config"], **(config_override or {})}
    state = {
        "question": question, "history": history or [], "persona": persona, "cfg": cfg,
        "variant": {"cot": True, "few_shot": True, "placement": "system", "instructions": "full",
                    "budget_warning": False, **(variant or {})},  # off: E5b showed no benefit
        "scratchpad": "", "steps": [], "pending": None, "final_answer": None, "iterations": 0,
        "format_errors": 0, "llm_calls": 0, "latency_s": 0.0, "tokens_out": 0, "ignored_params": [],
        "native_tool_calls": 0, "implicit_finals": 0, "ungrounded_finals": 0,
    }
    try:
        final = GRAPH.invoke(state, {"recursion_limit": 4 * MAX_STEPS + 5})
        error = None
    except Exception as exc:  # provider outage etc.: report it instead of crashing the app
        final = {**state, "final_answer": "Sorry, the model is unavailable right now. Please try again in a minute."}
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    result = {
        "error": error,
        "question": question, "persona": persona, "cfg": cfg, "variant": final["variant"],
        "final_answer": final["final_answer"], "steps": final["steps"],
        "tools_used": [s["tool"] for s in final["steps"] if s["type"] == "action"],
        "format_errors": final["format_errors"], "llm_calls": final["llm_calls"],
        "latency_s": round(final["latency_s"], 2), "tokens_out": final["tokens_out"],
        "ignored_params": final["ignored_params"], "native_tool_calls": final["native_tool_calls"],
        "implicit_finals": final.get("implicit_finals", 0), "ungrounded_finals": final.get("ungrounded_finals", 0),
        "scratchpad": final["scratchpad"],
    }
    if log:
        with open(LOG_DIR / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result},
                               ensure_ascii=False, default=str) + "\n")
    return result


def format_trace(result):
    """Readable Thought / Action / Observation trace."""
    out = []
    for s in result["steps"]:
        if s.get("thought"):
            out.append(f"Thought: {s['thought']}")
        if s["type"] == "action":
            out.append(f"Action: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)})")
            out.append(f"Observation: {json.dumps(s['observation'], ensure_ascii=False)[:400]}")
        elif s["type"] == "format_error":
            out.append(f"[format error] {s['error']}")
        else:
            out.append(f"Final Answer: {s['answer']}")
    return "\n".join(out)

# %% ui
MODEL_CHOICES = {
    "Qwen3.8 27B (Groq)": {"provider": "groq", "model": "qwen/qwen3.8-27b"},
    "Gemini 3.5 flash-lite (API)": {"provider": "gemini", "model": GEMINI_MODEL},
    "GPT-OSS 120B (Groq)": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "GPT-OSS 20B (Groq)": {"provider": "groq", "model": "openai/gpt-oss-20b"},
}


def build_ui():
    import gradio as gr

    persona_labels = {f"{p['name']} - {p['title']}": pid for pid, p in PERSONAS.items()}
    # Pre-select the configured default model (AGENT_MODEL) in the dropdown.
    default_label = next((k for k, v in MODEL_CHOICES.items() if v["model"] == DEFAULT_MODEL["model"]), list(MODEL_CHOICES)[0])

    def respond(message, history, persona_label, model_label):
        hist = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                 else " ".join(c.get("text", "") for c in m["content"] if isinstance(c, dict))}
                for m in history if isinstance(m, dict)][-8:]
        result = run_agent(message, persona=persona_labels[persona_label], history=hist, model=MODEL_CHOICES[model_label])
        trace = (f"**Persona:** {persona_label}  \n**Tools:** {', '.join(result['tools_used']) or 'none'}  \n"
                 f"**LLM calls:** {result['llm_calls']} | **format errors:** {result['format_errors']} | "
                 f"**latency:** {result['latency_s']} s\n\n```text\n{format_trace(result)}\n```")
        return result["final_answer"], trace

    with gr.Blocks(title="Cedar Trails") as demo:
        gr.Markdown("# Cedar Trails\nA ReAct hiking planner for the Lebanon Mountain Trail. Pick a persona and ask away.")
        with gr.Row():
            persona = gr.Dropdown(list(persona_labels), value=list(persona_labels)[0], label="Persona")
            model = gr.Dropdown(list(MODEL_CHOICES), value=default_label, label="Model")
        with gr.Row():
            with gr.Column(scale=3):
                trace = gr.Markdown(render=False)
                gr.ChatInterface(respond, additional_inputs=[persona, model], additional_outputs=[trace],
                                 examples=[["An easy day hike in the Shouf in October for two beginners?", list(persona_labels)[0], default_label],
                                           ["We plan Sir ed-Danniyeh to Ehden in January with a fit group. Go or no-go?", list(persona_labels)[1], default_label],
                                           ["What should I pack for 3 days on the trail in the north in August?", list(persona_labels)[2], default_label]],
                                 cache_examples=False)
            with gr.Column(scale=2):
                gr.Markdown("### ReAct trace")
                trace.render()
    return demo


if __name__ == "__main__":
    build_ui().launch()
