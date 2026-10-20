"""
Build the result tables (markdown) from scored runs under results/:

    results/<model>/<model>_<task_type>.json           raw adapter output (run_infer.py)
    results/<model>/<model>_<task_type>_results.json   per-record scores (evaluate.py)

Every model directory found under --results is included, in MODEL_ORDER first and
then alphabetically; task types without a scored file are skipped. Tables:

  1. Overall (macro-average over the task types present, +- 95% bootstrap CI)
  2. Strict Success by task type
  3. Error breakdown among failed samples
  4. Self-correction 2x2 (decision completeness x correction distance)

Clarify quality and result grounding are disabled in scorer.py, so neither
appears here.

Usage:
    python evaluation/make_tables.py [--results results] [--data data/final_data] [--out results/summary.md]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from scorer import _normalize

ROOT = Path(__file__).resolve().parent.parent

MODEL_ORDER = [("gpt_realtime", "GPT-Realtime"), ("gemini_live", "Gemini-Live"), ("grok_voice", "Grok-Voice"),
               ("nova_sonic", "Nova-Sonic"), ("step_audio2", "Step-Audio2"), ("voxmind", "VoxMind")]
FAMILIES = [
    ("Intent Understanding", ["call_or_not", "tool_selection_hard_negative", "parallel_tool_call"]),
    ("Information Extraction & Argument Binding",
     ["explicit_argument_binding", "resolvable_implicit_argument", "missing_argument"]),
    ("Result Reuse", ["sequential_tool_call"]),
    ("Self-Correction", ["tool_self_correction", "argument_self_correction"]),
]
TASKS = [t for _, ts in FAMILIES for t in ts]
METRICS = [("Strict Success", "strict"), ("Action Acc", "action"), ("Tool-F1", "tool_f1"),
           ("Schema Valid", "schema"), ("Arg Acc", "arg")]
ERROR_COLS = ["Action", "Tool", "Schema only", "Argument only"]

# Records with known data-quality problems in the full release; excluded from every table.
EXCLUDED_IDS = {
    "call_or_not_0001", "tool_selection_hard_negative_0009", "tool_selection_hard_negative_0026",
    "resolvable_implicit_argument_0029", "resolvable_implicit_argument_0060",
    "tool_self_correction_0025", "tool_self_correction_0027", "tool_self_correction_0028",
}


# ---------- loading ----------

def discover_models(results_dir):
    present = sorted(p.name for p in Path(results_dir).iterdir()
                     if p.is_dir() and any(p.glob(f"{p.name}_*_results.json")))
    known = [(k, n) for k, n in MODEL_ORDER if k in present]
    return known + [(k, k) for k in present if k not in dict(MODEL_ORDER)]


def load_scores(results_dir, key):
    by_task = {}
    for t in TASKS:
        path = Path(results_dir) / key / f"{key}_{t}_results.json"
        if path.exists():
            rows = json.loads(path.read_text())["scores"]
            by_task[t] = [s for s in rows if s["id"] not in EXCLUDED_IDS]
    return by_task


# ---------- metrics ----------

def metric_value(metric, s):
    """Per-record value of a metric, or None when the record is outside its denominator."""
    needs_call = s["tool_selection"] is not None
    if metric == "strict":
        return float(s["strict_success"])
    if metric == "action":
        return float(s["action_correct"])
    if metric == "tool_f1":
        return s["tool_selection"]["f1"] if needs_call else None
    if metric == "schema":
        return s.get("schema_valid_when_called")
    if metric == "arg":
        return s["argument_accuracy"] if needs_call else None
    raise ValueError(metric)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def task_mean(rows, metric):
    return mean(metric_value(metric, s) for s in rows) if rows else None


def macro(by_task, metric):
    return mean(task_mean(rows, metric) for rows in by_task.values())


def bootstrap_ci(by_task, metric, n_boot=1000, seed=0):
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        resampled = {t: [rng.choice(rows) for _ in rows] for t, rows in by_task.items() if rows}
        v = macro(resampled, metric)
        if v is not None:
            stats.append(v)
    if not stats:
        return None
    stats.sort()
    lo, hi = stats[int(0.025 * len(stats))], stats[max(int(0.975 * len(stats)) - 1, 0)]
    return (hi - lo) / 2


def error_shares(rows):
    """Share of failed samples per column (Action / Tool overlap; Schema only and
    Argument only are mutually exclusive with them and with each other)."""
    failed = [s for s in rows if not s["strict_success"]]
    counts = [0, 0, 0, 0]
    for s in failed:
        action_bad = not s["action_correct"]
        tool_bad = s["tool_selection"] is not None and s["tool_selection"]["f1"] < 1.0
        schema_bad = s["schema_valid"] is not None and s["schema_valid"] < 1.0
        arg_bad = s["argument_accuracy"] is not None and s["argument_accuracy"] < 1.0
        counts[0] += action_bad
        counts[1] += tool_bad
        if not action_bad and not tool_bad:
            if schema_bad:
                counts[2] += 1
            elif arg_bad:
                counts[3] += 1
    return len(failed), [100 * c / len(failed) if failed else 0.0 for c in counts]


# ---------- self-correction 2x2 ----------

def _contains_obsolete(value, obsolete):
    nv, no = _normalize(value), _normalize(obsolete)
    if nv == no:
        return True
    if isinstance(nv, list):
        return any(x == no or (isinstance(x, str) and isinstance(no, str) and no and no in x) for x in nv)
    return isinstance(nv, str) and isinstance(no, str) and bool(no) and no in nv


_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}


def _spoken_to_value(text):
    """'four' -> 4, 'E V dash four twelve' -> 'EV-412'; other text unchanged."""
    words = text.split()
    if len(words) == 1 and words[0].lower() in _NUM_WORDS:
        return _NUM_WORDS[words[0].lower()]
    if any(w.lower() in _NUM_WORDS or w.lower() == "dash" for w in words) and len(words) > 1:
        parts = []
        for w in words:
            lw = w.lower()
            parts.append("-" if lw == "dash" else str(_NUM_WORDS[lw]) if lw in _NUM_WORDS else w)
        return "".join(parts)
    return text


def obsolete_values(item):
    """{corrected argument: pre-correction value}. Structured initial_intent when
    the data has it, otherwise the obsolete_value span (spoken form converted)."""
    ann = item["annotations"]
    keys = ann.get("corrected_arguments", [])
    init = ann.get("initial_intent")
    if isinstance(init, dict):
        return {k: init["arguments"][k] for k in keys if k in init.get("arguments", {})}
    spans = [s["text"] for s in ann.get("correction_spans", []) if s["type"] == "obsolete_value"]
    return {keys[0]: _spoken_to_value(spans[0])} if keys and spans else {}


def failed_to_update(task, item, record):
    """Automatic failure-to-update check over ALL tool calls the model made.
    Tool level: the obsolete tool was called. Argument level: a call to the
    final tool has a corrected argument equal to / containing the obsolete
    value. Samples with no tool call are not counted."""
    ann = item["annotations"]
    calls = record.get("actual_tool_calls") or []
    if task == "tool_self_correction":
        return any(c.get("name") == ann.get("obsolete_tool") for c in calls)
    obsolete = obsolete_values(item)
    final_tool = item["gold"].get("tool")
    for c in calls:
        if c.get("name") != final_tool:
            continue
        args = c.get("args") or {}
        for k, old in obsolete.items():
            if k in args and _contains_obsolete(args[k], old):
                return True
    return False


def self_correction_cells(results_dir, data_dir, key):
    """{(task, completeness, distance): (n, strict%, failure-to-update%)}"""
    out = {}
    for task in ["argument_self_correction", "tool_self_correction"]:
        scores_path = Path(results_dir) / key / f"{key}_{task}_results.json"
        raw_path = Path(results_dir) / key / f"{key}_{task}.json"
        data_path = Path(data_dir) / f"{task}.json"
        if not (scores_path.exists() and raw_path.exists() and data_path.exists()):
            continue
        data = {d["id"]: d for d in json.loads(data_path.read_text())}
        recs = {r["id"]: r for r in json.loads(raw_path.read_text())}
        cells = defaultdict(list)
        for s in json.loads(scores_path.read_text())["scores"]:
            if s["id"] in EXCLUDED_IDS or s["id"] not in data or s["id"] not in recs:
                continue
            ann = data[s["id"]]["annotations"]
            if "completeness" not in ann or "distance_bucket" not in ann:
                continue
            cells[(ann["completeness"], ann["distance_bucket"])].append(
                (s["strict_success"], failed_to_update(task, data[s["id"]], recs[s["id"]])))
        for (comp, dist), v in cells.items():
            out[(task, comp, dist)] = (len(v), 100 * sum(a for a, _ in v) / len(v),
                                       100 * sum(b for _, b in v) / len(v))
    return out


# ---------- rendering ----------

def fmt(v):
    return "—" if v is None else f"{v:.3f}"


def bold_best(cells, values):
    vals = [v for v in values if v is not None]
    if not vals:
        return list(cells)
    best = round(max(vals), 3)
    return [f"**{c}**" if v is not None and round(v, 3) == best else c for c, v in zip(cells, values)]


def family_rows(n_cols, render, tasks_present):
    lines = []
    for family, tasks in FAMILIES:
        tasks = [t for t in tasks if t in tasks_present]
        if tasks:
            lines.append(f"| ***{family}*** |" + " |" * n_cols)
            lines.extend(render(t) for t in tasks)
    return lines


def two_by_two(name, cells, idx):
    cols = [("argument_self_correction", "short"), ("argument_self_correction", "long"),
            ("tool_self_correction", "short"), ("tool_self_correction", "long")]
    lines = [f"| {name} | Arg: Low | Arg: High | Tool: Low | Tool: High |", "|---|---|---|---|---|"]
    for dist, label in [("short", "Short distance"), ("long", "Long distance")]:
        row = [f"{cells[(t, c, dist)][idx]:.1f}% (n={cells[(t, c, dist)][0]})" if (t, c, dist) in cells else "—"
               for t, c in cols]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--data", default=str(ROOT / "data" / "final_data"))
    p.add_argument("--out", default=None, help="defaults to <results>/summary.md")
    args = p.parse_args()
    out_path = Path(args.out or Path(args.results) / "summary.md")

    model_keys = discover_models(args.results)
    if not model_keys:
        raise SystemExit(f"no scored runs (<model>/<model>_<task>_results.json) under {args.results}")
    models = {name: load_scores(args.results, key) for key, name in model_keys}
    names = [n for _, n in model_keys]
    tasks_present = [t for t in TASKS if any(t in models[n] for n in names)]

    header = "| {} | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)

    out = ["# Results", "",
           f"Models: {', '.join(names)}. Task types: {len(tasks_present)}. "
           f"{len(EXCLUDED_IDS)} records with known data issues are excluded. "
           "Clarify quality and result grounding are disabled in `evaluation/scorer.py`.", "",
           f"## Table 1. Overall (macro-average over {len(tasks_present)} task types, ± 95% CI)", "",
           header.format("Metric"), sep]
    for label, m in METRICS:
        values = [macro(models[n], m) for n in names]
        cells = bold_best([fmt(v) for v in values], values)
        cis = [bootstrap_ci(models[n], m) for n in names]
        cells = [f"{c} ±{ci:.3f}" if ci is not None else c for c, ci in zip(cells, cis)]
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    out += ["", "## Table 2. Strict Success by task", "", header.format("Task"), sep]

    def strict_row(t):
        values = [task_mean(models[n].get(t), "strict") for n in names]
        cells = [f"{fmt(v)} (n={len(models[n].get(t) or [])})" for v, n in zip(values, names)]
        return f"| {t} | " + " | ".join(bold_best(cells, values)) + " |"
    out += family_rows(len(names), strict_row, tasks_present)

    out += ["", "## Table 3. Error breakdown among failed samples (% of failures)", "",
            "- **Action** / **Tool**: the action / tool selection is wrong (a sample can be wrong at both).",
            "- **Schema only**: action and tool are correct, but a required argument is missing or malformed.",
            "- **Argument only**: action, tool and schema are all correct; only the argument values are wrong."]
    for i, n in enumerate(names):
        out += ["", f"### Table 3{chr(97 + i)}. {n}", "",
                "| Task (failed / n) | " + " | ".join(ERROR_COLS) + " |", "|---|" + "---|" * len(ERROR_COLS)]

        def error_row(t):
            rows = models[n].get(t) or []
            n_failed, values = error_shares(rows)
            return f"| {t} ({n_failed} / {len(rows)}) | " + " | ".join(
                "0" if v == 0 else f"{v:.1f}%" for v in values) + " |"
        out += family_rows(len(ERROR_COLS), error_row, tasks_present)

    sc = {n: self_correction_cells(args.results, args.data, key) for key, n in model_keys}
    if any(sc.values()):
        out += ["", "## Table 4. Self-correction 2x2 (decision completeness × correction distance)", "",
                "Low/High = `completeness` (short/long: how complete the original request was before the "
                "correction); Short/Long distance = `distance_bucket`.", "",
                "### Table 4a. Strict Success (%, higher is better)", ""]
        for n in names:
            out += two_by_two(n, sc[n], 1) + [""]
        out += ["### Table 4b. Failure-to-Update Rate (%, lower is better)", "",
                "Counts samples where any tool call kept the retracted intent: the obsolete tool was called "
                "(tool level), or the final tool was called with a corrected argument still holding the old "
                "value (argument level). Samples with no tool call are not counted.", ""]
        for n in names:
            out += two_by_two(n, sc[n], 2) + [""]

    md = "\n".join(out).rstrip() + "\n"
    out_path.write_text(md)
    print(md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
