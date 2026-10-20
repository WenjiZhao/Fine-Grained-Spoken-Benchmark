"""OpenAI Realtime API (gpt-realtime) adapter: dataset record -> normalized result dict.

Talks directly to wss://api.openai.com/v1/realtime over the raw event
protocol (no OpenAI SDK realtime client dependency, same reasoning as
models/step_audio2/adapter.py picking the in-process route over standing up
extra infra: one fewer moving part with its own version-compat surface).
Session config sets output_modalities=["text"] -- same choice models/voxmind/
adapter.py and models/voicechat/adapter.py already made (response_mode="text"
/ decode_audio=False): evaluation/scorer.py never scores audio, so there is
no reason to pay for speech synthesis on the output side. The INPUT side is
still genuinely full-duplex real-time audio -- see models/common/
duplex_audio.py's docstring for why that half of the pipeline can't be
simplified the same way and must stream resolved_audio_path(record,
duplex="full") frame-by-frame in real time instead of uploading it whole.

This targets the GA (non-beta) wire shape -- no `OpenAI-Beta` header,
`session.type: "realtime"`, `output_modalities` instead of `modalities`, and
audio config nested under `session.audio.input`/`session.audio.output`
instead of flat `input_audio_format`/`turn_detection` fields. An earlier
version of this adapter used the older beta shape and got a live
`invalid_request_error.beta_api_shape_disabled` close (code 4000) back from
a real session -- confirmed against https://developers.openai.com/api/
reference/resources/realtime/client-events and .../server-events before
switching to this shape; see the CAVEAT below for what's still unverified.

Turn-taking mechanics, matched to this benchmark's shape:
  - turn_detection: server_vad is left on, so the model decides on its own
    when the user's simulated turn has ended (the trailing silence in
    full_duplex_audio_paths -- see data_synthetic/tts.py -- is what it's
    reacting to) and auto-creates its first response, exactly like a live
    caller. No input_audio_buffer.commit / response.create call is made for
    turn 1.
  - After that, function-calling is a fixed-point loop, same shape as every
    other adapter here: a response.done event's response.output may contain
    one or more function_call items; each gets a function_call_output item
    fed back and a response.create is sent to keep going, until a
    response.done comes back with no function_call items left (that
    response's text completes the final_response, which joins the text of every
    completed round) or MAX_STEPS is hit.
  - Every function_call item the server names in a response gets answered
    (never silently skipped), even a repeat of an already-made call --
    dropping an unanswered call_id in gpt-realtime's protocol is not
    something this adapter wants to speculate is safe. A repeat reuses the
    cached result from MockToolExecutor's first real call instead of
    re-invoking it (calling a DB-backed tool twice would double-apply its
    side effect, e.g. add_to_cart appending a second cart line) -- only the
    first occurrence of a given (name, args) pair is counted in
    actual_tool_calls, matching every other adapter's dedup-for-scoring
    convention.

VERIFIED (2026-09-30) end to end against a live session (see README.md for the
full account): a real explicit_argument_binding record produced an exact-match
gold tool call and a grounded final answer, through exactly the round shape
this adapter is built to handle -- several `status: "cancelled"` rounds (real
full-duplex audio isn't digitally silent; ambient noise plus the transcript's
own disfluency pauses retrigger server_vad mid-response), a function_call
round, then a completed text round. Two real fixes came out of that run and
are already reflected below: the GA (non-beta) session shape (`session.type`,
`output_modalities`, `session.audio.input`), and continuing past a cancelled
response instead of treating its empty output as the final answer. What's
still unverified: parallel/multiple function calls in a single round, and any
session-level error path that one run didn't happen to trigger. Recheck
https://developers.openai.com/api/reference/resources/realtime/client-events
/ .../server-events if OpenAI revises the GA shape again.
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import websockets

BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from models.common.dataset import resolved_audio_path, task_type_system_suffix, tool_schemas_for  # noqa: E402
from models.common.duplex_audio import load_duplex_stream  # noqa: E402
from models.common.result_schema import build_result_record, normalize_calls  # noqa: E402
from tools.executor import MockToolExecutor  # noqa: E402

REALTIME_URL = "wss://api.openai.com/v1/realtime"
SAMPLE_RATE = 24000  # pcm16 is fixed at 24kHz mono, both directions, per OpenAI's docs
FRAME_MS = 32

# Upper bound on response.create rounds per record (mirrors MAX_STEPS in every other
# adapter here) -- a safety cap only, the loop normally stops as soon as a response
# comes back with no function_call items left.
MAX_STEPS = 8

# How long to wait for the NEXT response.done/error after issuing a request (turn 1's
# own request is server_vad detecting end-of-speech from the streamed audio, not
# something this adapter sends) before giving up on the record instead of hanging a
# whole run over one bad connection.
RESPONSE_TIMEOUT_S = 60.0

SYSTEM_MESSAGE = (
    "You are a helpful voice assistant. You understand spoken requests and decide whether "
    "to answer directly from your own knowledge or call your available tools.\n\n"
    "Tool Decision Rules:\n"
    "- If the request can be answered directly, answer normally -- do not call any tool.\n"
    "- If the request needs a capability listed in your available tools, call that tool "
    "instead of guessing the answer.\n"
    "- Never invent or call a tool name that is not literally one of your available tools.\n"
    "- Tool arguments must be values the user actually said or values returned by an "
    "earlier tool result; if a required argument is missing from both, ask the user "
    "instead of guessing.\n"
    "- If a tool call fails or returns an error, do not retry it for the same request -- "
    "tell the user the tool has an issue.\n"
    "- If the request needs several tools in sequence (e.g. search first, then act on a "
    "result), call them one after another: after each tool result comes back, call the "
    "next tool with the values from that result before answering the user."
)


def _openai_realtime_tool(schema: dict) -> dict:
    """tools/tool_schemas.json's shape -> the Realtime session.update event's own tool
    object shape -- {"type": "function", "name", "description", "parameters"}, FLAT, no
    nested "function" key. That nesting is the Chat Completions / Responses API
    convention; session.update's own "tools" array is flat (see
    https://platform.openai.com/docs/guides/realtime-conversations#function-calling).
    Do not copy models/step_audio2 or models/voxmind's _openai_tool_schema() here --
    those build the *nested* shape for a different (in-context, non-API) purpose."""
    return {
        "type": "function",
        "name": schema["name"],
        "description": schema.get("description", schema["name"]),
        "parameters": schema.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }


def _call_key(name: str, arguments: dict) -> Tuple[str, str]:
    return name, json.dumps(arguments or {}, sort_keys=True)


class GPTRealtimeAdapter:
    def __init__(self, model: str = "gpt-realtime", api_key: Optional[str] = None, verbose: bool = False):
        self.model = model
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.verbose = verbose

    def run(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        """Sync entry point for run_infer.py's sequential per-record loop. A batch
        script that wants several records running concurrently should await arun()
        directly instead (asyncio.run() can't nest inside an already-running loop)."""
        return asyncio.run(self.arun(record, all_schemas))

    async def arun(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        tools_schema = [_openai_realtime_tool(s) for s in tool_schemas_for(record, all_schemas)]
        system_message = SYSTEM_MESSAGE + task_type_system_suffix(record)
        audio_path = resolved_audio_path(record, duplex="full")
        frames, frame_seconds = load_duplex_stream(audio_path, SAMPLE_RATE, FRAME_MS)

        executor = MockToolExecutor()  # fresh per record, per executor.py's own contract
        raw_calls: List[Dict[str, Any]] = []
        seen_calls = set()
        call_result_cache: Dict[Tuple[str, str], dict] = {}
        raw_generation: Dict[str, Any] = {}
        final_response = ""

        url = f"{REALTIME_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "instructions": system_message,
                    "audio": {"input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "turn_detection": {"type": "server_vad"},
                    }},
                    **({"tools": tools_schema, "tool_choice": "auto"} if tools_schema else {}),
                },
            }))
            await self._await_session_updated(ws)

            send_task = asyncio.create_task(self._send_audio(ws, frames, frame_seconds, self.verbose))
            try:
                for step in range(1, MAX_STEPS + 1):
                    event = await asyncio.wait_for(self._next_response_event(ws, self.verbose), RESPONSE_TIMEOUT_S)
                    if event["type"] == "error":
                        raw_generation[f"turn{step}_error"] = event.get("error")
                        break

                    response = event.get("response", {})
                    raw_generation[f"turn{step}"] = response
                    text_parts = []
                    function_calls = []
                    for item in response.get("output", []) or []:
                        if item.get("type") == "message":
                            for content in item.get("content", []) or []:
                                text = content.get("text") or content.get("transcript")
                                if content.get("type") in ("text", "output_text") and text:
                                    text_parts.append(text)
                        elif item.get("type") == "function_call":
                            function_calls.append(item)
                    if response.get("status") == "cancelled":
                        # A function_call in a cancelled response was cut off
                        # mid-generation (its arguments JSON is truncated) and was
                        # never committed -- don't execute or score it; the
                        # completed retry server_vad triggers next is the real call.
                        continue

                    if text_parts:
                        # Accumulate across rounds -- text spoken alongside a function
                        # call is part of the answer the user heard (same as
                        # gemini_live's text_parts).
                        final_response = f"{final_response} {''.join(text_parts)}".strip()

                    if not function_calls:
                        if response.get("status") == "cancelled":
                            # server_vad re-detected activity mid-response (real
                            # full-duplex audio is never perfectly silent -- ambient
                            # noise in the recording can retrigger it) and cut this
                            # response off before it produced anything. That is not
                            # this record's grounded answer -- server_vad's own
                            # create_response default means another response.done is
                            # still coming once whatever it just detected quiets back
                            # down, so keep waiting for it instead of ending the
                            # record on an empty non-answer.
                            continue
                        break  # a genuinely completed text-only turn is the grounded final answer

                    for item in function_calls:
                        name = item.get("name", "")
                        try:
                            arguments = json.loads(item.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        key = _call_key(name, arguments)
                        if key not in call_result_cache:
                            call_result_cache[key] = executor.call(name, **arguments)
                        if key not in seen_calls:
                            seen_calls.add(key)
                            raw_calls.append({"name": name, "arguments": arguments})
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": item.get("call_id"),
                                "output": json.dumps(call_result_cache[key], ensure_ascii=False),
                            },
                        }))
                    await ws.send(json.dumps({"type": "response.create"}))
            finally:
                send_task.cancel()

        return build_result_record(
            record, normalize_calls(raw_calls), executor.tool_results(), final_response, raw_generation
        )

    @staticmethod
    async def _send_audio(ws, frames: List[bytes], frame_seconds: float, verbose: bool) -> None:
        try:
            for frame in frames:
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(frame).decode("ascii"),
                }))
                await asyncio.sleep(frame_seconds)
            if verbose:
                print("  [gpt_realtime] finished streaming input audio")
        except asyncio.CancelledError:
            pass

    @staticmethod
    async def _await_session_updated(ws) -> None:
        while True:
            event = json.loads(await ws.recv())
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise RuntimeError(f"OpenAI Realtime session.update rejected: {event.get('error')}")

    @staticmethod
    async def _next_response_event(ws, verbose: bool) -> dict:
        """Drain events until the next response.done or error -- everything else
        (session bookkeeping, speech_started/stopped, per-item/delta streaming events)
        is irrelevant here since modalities=["text"] means response.done's own
        response.output already carries the complete, fully-assembled turn."""
        while True:
            event = json.loads(await ws.recv())
            etype = event.get("type")
            if verbose:
                print(f"  [gpt_realtime] <- {etype}")
            if etype in ("response.done", "error"):
                return event
