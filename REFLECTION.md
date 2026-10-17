All numbers come from `results/runs.jsonl` (128 runs) and `results/judge.jsonl` (LLM judge, 1-5). "Quality" is the mean of
four automatic checks: completed, followed the output format, used the required tools, and stated the checkable facts from the observations.

### 1. Which persona gave the most helpful or natural results?
**Nour, the Gear Quartermaster.** Across all completed runs Nour had the best judge scores (persona fit **3.95**, helpfulness
**4.36**), and on the best model (Qwen3.8-27B) Nour scored 4.5 / 5.0 with perfect quality. Packing is a well-structured task:
one tool returns most of what is needed, and the persona's "one line per item, no vague advice" style matches what a hiker wants.
**Rami** was a close second on helpfulness (4.30) and had the highest automatic quality (0.91), but his persona fit varied with
temperature: the judge rated him 3.0-3.3 at temperature 0.4-0.8 and **4.67 at 1.2**, where his descriptions became more vivid.
**Hala** was the hardest persona (fit 3.33): a strict, rule-based verdict exposes every weakness of a model. Only Qwen got all of
her go / no-go verdicts right; Gemini, GPT-OSS-120B and GPT-OSS-20B were right 0-50% of the time.

### 2. Which prompt and configuration combination performed best?
**Qwen3.8-27B + the full instructions block in the system prompt (with the worked example and `Thought:` lines) + each persona's own
low-to-moderate temperature** (E1 quality 0.88, the only model with zero format errors and 100% correct verdicts). Supporting evidence:
* **Model choice mattered most** (E1): Qwen 0.88 > GPT-OSS-20B 0.78 > Gemini 3.5 flash-lite 0.53 > GPT-OSS-120B 0.33. The biggest
  model was the worst ReAct engine here, because it is trained for its own native tool-calling format and kept ignoring ours.
* **System-prompt instructions beat in-message instructions** (E3: 0.70 vs 0.44 quality, 0.08 vs 1.08 format errors per run).
* **Few-shot did not help GPT-OSS-20B** (E3: 0.48 vs 0.66). Combined with in-message placement it was the worst setting (0.08-0.12):
  the model copied the example's `Action:` lines into its native tool channel and ran out of steps.
* **Visible chain-of-thought made no measurable difference** on GPT-OSS-20B (0.55 vs 0.60), which already reasons on a hidden
  channel. We keep `Thought:` lines anyway because they make every trace explainable.
* **Temperature** (E4, Rami on Qwen): quality stayed at about 1.0 from 0.0 to 1.2 while answer diversity rose from 0.39 to 0.72
  and persona fit peaked at 1.2. At **1.6 the agent collapsed** (quality 0.21, 4 format errors per run, garbled text).
  **top_p 0.3** gave the least diverse answers (0.29) with full quality: a good fit for Hala. **Presence and frequency penalties
  of 1.5** kept quality and raised diversity (0.68) and helpfulness (4.7-5.0): useful for Rami and Nour. **max_tokens 150**
  still worked because each ReAct step is short, but answers shrank to 58 words and persona fit fell to 3.0.
* **top_k** (Gemini only) had no visible effect: even top_k = 1 was not deterministic (diversity 0.55), because Gemini's hidden
  thinking runs before sampling.
* **Persona configs on a weak model** (E2, GPT-OSS-120B, 2 runs per cell) were dominated by noise: low-temperature configs helped
  every persona slightly, so a reliable model matters more than fine-tuning sampling parameters.

### 3. How well did the agent reason and use tools?
With Qwen, very well: it searched, then estimated or checked safety, and cited the observed numbers (e.g. "max altitude 2000 m,
above the 1500 m threshold, January snow risk high, danger alert: NO-GO"). Tool recall was 100% in E1. Failure modes we observed:
* **Step-limit exhaustion** from repeated calls: Qwen searched 4 times before estimating on an open-ended request, and GPT-OSS-20B
  called `check_safety_alerts` four times in a row. A last-step "give the final answer now" warning (E5b) **did not fix this**,
  because the steps were already wasted in a loop. The right fix is loop detection (reject an identical repeated call), left as future work.
* **Hallucinated tools and observations**: GPT-OSS-120B called `repo_browser.open_file` (a tool from its training data), wrote
  "Observation: a danger alert..." without calling anything, and once invented sections "S22" and "S19". Our **grounding guard**
  (no final answer before any tool call) caught these, but at the step limit it still produced a wrong NO-GO for a safe section.
* **Format drift**: GPT-OSS often wrote a perfect answer without the `Final Answer:` prefix, or used its native tool channel.
  We accept both (implicit finals; native calls converted back to text Actions) and count them.

### 4. Did the instructions block change the agent's behaviour?
**Yes, strongly.** In E5a we replaced the dedicated block with a one-line minimum (tool list plus action syntax) on Qwen:
quality fell from **0.88 to 0.58**, format compliance from **83% to 0%**, and Hala's January safety check went from a perfect
NO-GO citing the alert, the altitude and the rule to running out of steps without any verdict. The block's specific parts
matched specific behaviours: the **decision rule** produced consistent verdicts, the **grounding rule** made the agent quote
observation numbers instead of guessing, and the **output formats** made answers easy to scan and to evaluate automatically.
The LLM judge rated tone and helpfulness about the same with and without the block (3.2 / 3.8 for both), which shows that the
block mainly buys **reliability and structure**, while tone comes from the persona.

### 5. Biggest implementation challenges
1. **Every provider bends the text-ReAct protocol differently.** GPT-OSS emitted native tool calls that Groq rejected
   ("Tool choice is none, but model called a tool"), or output Groq could not parse. Gemini 3.5 flash-lite returned empty
   `MALFORMED_RESPONSE` replies on multi-step prompts, counts hidden thinking tokens against `max_output_tokens`, and rejects
   presence / frequency penalties outright.
2. **Our own harness first under-reported the models.** The first experiment pass marked 56 runs as failures largely because the
   parser was too strict (answers without the `Final Answer:` prefix). We fixed the parser, added the grounding guard, and re-ran
   those runs; the originals are kept in `results/*_discarded_v1.jsonl` for transparency.
3. **Free-tier limits shaped the design**: 15 requests/min on Gemini, 8k tokens/min and 200k tokens/day per model on Groq. We
   ran one worker thread per model, made the runner resumable, and capped retry waits after a long `retry-after` stalled a run.
   Two E5b runs were skipped when Qwen's daily quota ran out.
4. **Measuring "good"**: we combined automatic checks (format, tools, facts, verdicts) with an LLM judge for tone and helpfulness.
   With 2-3 repeats per setting, small differences (for example in E2) are within noise; the large effects (model choice,
   instruction placement, the instructions block, temperature 1.6) are the robust findings.
