"""Shared dataset loading for every model adapter.

Every adapter consumes the same data/final_data/<task_type>.json files and the
same tools/tool_schemas.json -- this module is the one place both are read
from, so a schema/path change only has to be made here.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

BENCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = BENCH_ROOT / "data"
DEFAULT_DATASET_PATH = DEFAULT_DATA_ROOT / "final_data"  # a directory: load_dataset() loads every <task_type>.json in it
DEFAULT_SCHEMAS_PATH = BENCH_ROOT / "tools" / "tool_schemas.json"

# Appended to a model's system prompt for records of the matching task_type. Used by
# the closed-source realtime adapters (gpt_realtime, gemini_live, grok_voice,
# nova_sonic); the open-source adapters (voxmind, step_audio2) do not append it.
# Each rule targets a specific failure mode observed for that task_type.
TASK_TYPE_SYSTEM_SUFFIXES: Dict[str, str] = {
    # call_or_not: probes whether the model calls a tool only when it directly
    # satisfies the request rather than as a preparatory/exploratory step.
    "call_or_not": (
        "\n\nAdditional rule for this request: select a tool only if it can directly "
        "satisfy the user's current request. Do not call a tool merely as a preparatory, "
        "exploratory, or prerequisite step. If no available tool can directly fulfill "
        "the request, do not call any tool -- tell the user you cannot help with that."
    ),
    # missing_argument: the user leaves a required argument unstated; both adapters were
    # observed guessing/inventing a value (or using a placeholder like "today"/"not set")
    # instead of asking, in ~93% of these records.
    "missing_argument": (
        "\n\nAdditional rule for this request: if a required argument for the tool you "
        "want to call is not stated, or only vaguely stated, by the user, do not guess, "
        "invent, or use a placeholder value for it. Ask the user a clarifying question "
        "for that missing value instead of calling the tool."
    ),
    # resolvable_implicit_argument: the user refers to a value implicitly ("near me",
    # "today", "my usual card"); both adapters were observed either using that literal
    # wording as the argument, guessing a concrete value from their own knowledge, or
    # inventing a nonexistent merged tool name, instead of first calling the dedicated
    # resolver tool (get_current_location / get_current_date / get_current_time /
    # get_user_profile) and chaining its result into the real tool call.
    "resolvable_implicit_argument": (
        "\n\nAdditional rule for this request: if the user refers to a value implicitly "
        "(e.g. 'my current location', 'today', 'my usual card') instead of stating it "
        "directly, you need to find a way to first resolve information like the current "
        "location, date/time, or user profile details, and use that resolved result as "
        "the argument in your next tool call. Do not use the user's literal wording or a "
        "guessed value as the argument, and do not invent a tool name that is not "
        "literally one of your available tools."
    ),
}


def task_type_system_suffix(record: Dict[str, Any]) -> str:
    """The extra system-prompt text (if any) for record["task_type"], or "" if none."""
    return TASK_TYPE_SYSTEM_SUFFIXES.get(record.get("task_type"), "")


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load a single data/final_data/<task_type>.json file, or every
    <task_type>.json directly inside a directory (sorted for a stable,
    reproducible record order)."""
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text())

    records: List[Dict[str, Any]] = []
    for f in sorted(path.glob("*.json")):
        records.extend(json.loads(f.read_text()))
    return records


def load_tool_schemas(path: Path = DEFAULT_SCHEMAS_PATH) -> Dict[str, dict]:
    """name -> full schema dict (parameters, etc.), from tools/tool_schemas.json."""
    return {t["name"]: t for t in json.loads(Path(path).read_text())}


def tool_schemas_for(record: Dict[str, Any], all_schemas: Dict[str, dict]) -> List[dict]:
    """Resolve record["available_tools"] (name-only) to full parameter schemas.

    A name not present in tools/tool_schemas.json is skipped with a printed
    warning rather than raising -- data curation issues shouldn't crash a
    full inference run partway through.
    """
    resolved = []
    for entry in record.get("available_tools", []):
        name = entry.get("name") if isinstance(entry, dict) else entry
        schema = all_schemas.get(name)
        if schema is None:
            # data/final_data names the dotted seqdep tools with underscores
            # ("items_search" for "items.search") since OpenAI function names
            # can't contain dots -- expose the schema under the record's name.
            dotted = next((n for n in all_schemas if "." in n and n.replace(".", "_") == name), None)
            if dotted is not None:
                schema = {**all_schemas[dotted], "name": name}
        if schema is None:
            # Same idea for the mixed-case W2C tools, which data/final_data now
            # names in snake_case ("events_3_find_events" for "Events_3_FindEvents").
            squash = lambda n: n.replace("_", "").lower()
            mixed = next((n for n in all_schemas if n != n.lower() and squash(n) == squash(name)), None)
            if mixed is not None:
                schema = {**all_schemas[mixed], "name": name}
        if schema is None:
            print(f"[dataset] warning: record {record.get('id')} references unknown tool {name!r}")
            continue
        resolved.append(schema)
    return resolved


def available_tool_names(record: Dict[str, Any]) -> List[str]:
    """record["available_tools"] as a plain name list -- pass to
    MockToolExecutor(allowed_tools=...) so a call to a tool the model was not
    offered for this record fails like an unknown tool instead of executing."""
    return [
        entry.get("name") if isinstance(entry, dict) else entry
        for entry in record.get("available_tools", [])
    ]


def resolved_audio_path(record: Dict[str, Any], duplex: str = "half") -> Path:
    """The (first) local audio file for a record, as an absolute path.

    data/final_data/<task_type>.json records carry both half_duplex_audio_paths
    and full_duplex_audio_paths (paths already rooted at BENCH_ROOT, e.g.
    "data/final_data/half_duplex/call_or_not_audio/call_or_not_0001.wav" /
    ".../full_duplex/..."); `duplex` selects which ("half" or "full", default
    "half" to preserve every existing caller's behavior). Records that carry
    resolved_audio_paths instead have no duplex distinction and fall back to
    that key regardless of `duplex`.
    """
    if duplex not in ("half", "full"):
        raise ValueError(f"duplex must be 'half' or 'full', got {duplex!r}")
    key = f"{duplex}_duplex_audio_paths"
    paths = record.get(key) or record.get("resolved_audio_paths") or []
    if not paths:
        raise ValueError(f"record {record.get('id')} has no {key} or resolved_audio_paths")
    return BENCH_ROOT / paths[0]
