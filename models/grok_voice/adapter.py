"""xAI Grok Voice Agent API adapter: dataset record -> normalized result dict.

Talks directly to wss://api.x.ai/v1/realtime over the raw event protocol, same
approach as models/gpt_realtime/adapter.py -- xAI's Voice Agent API is an
OpenAI-Realtime-compatible event protocol (session.update /
input_audio_buffer.append / conversation.item.create / response.create /
response.done), so this adapter mirrors gpt_realtime's structure almost
one-for-one. Input is resolved_audio_path(record, duplex="full") streamed
frame-by-frame in real time (see models/common/duplex_audio.py).

Differences from gpt_realtime, per https://docs.x.ai/docs/guides/voice/agent:
  - session.update has no `type: "realtime"` / `output_modalities` field and
    no documented text-only output mode, so output stays audio and the
    final_response is read from the output audio transcript (same trick as
    models/gemini_live's output_audio_transcription). Audio bytes are never
    kept or decoded.
  - `turn_detection` sits at the session top level, not under
    session.audio.input.
  - The docs only document function calls via the streaming
    `response.function_call_arguments.done` event (name, call_id, arguments),
    not response.done's payload. So this adapter collects function calls and
    transcript text from the streaming events of each response, and also
    merges anything response.done's response.output carries (deduped by
    call_id) in case the server does fill it in like OpenAI's.

Turn-taking / tool-loop mechanics are identical to gpt_realtime: server_vad
auto-creates the first response; each round's function calls all get a
function_call_output, then one response.create; a cancelled response's calls
are dropped and the loop waits for the server's retry; repeats of an
already-made (name, args) call reuse the cached result and are not
double-counted in actual_tool_calls.
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

REALTIME_URL = "wss://api.x.ai/v1/realtime"
DEFAULT_MODEL = "grok-voice-latest"
DEFAULT_VOICE = "eve"
SAMPLE_RATE = 24000
FRAME_MS = 32

MAX_STEPS = 8
RESPONSE_TIMEOUT_S = 60.0

# Transcript/text "done" events across naming variants (xAI follows OpenAI's GA
# names; the beta names are accepted too in case either shows up).
TEXT_DONE_EVENTS = {
    "response.output_audio_transcript.done": "transcript",
    "response.audio_transcript.done": "transcript",
    "response.output_text.done": "text",
    "response.text.done": "text",
}

# Same system prompt as gpt_realtime/gemini_live so the comparison is fair.
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


def _grok_tool(schema: dict) -> dict:
    """tools/tool_schemas.json's shape -> the flat {"type": "function", name, description,
    parameters} tool object session.update takes (same flat shape as gpt_realtime)."""
    return {
        "type": "function",
        "name": schema["name"],
        "description": schema.get("description", schema["name"]),
        "parameters": schema.get("parameters", {"type": "object", "properties": {}, "required": []}),
    }


def _call_key(name: str, arguments: dict) -> Tuple[str, str]:
    return name, json.dumps(arguments or {}, sort_keys=True)


class GrokVoiceAdapter:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 voice: str = DEFAULT_VOICE, verbose: bool = False):
        self.model = model
        self.api_key = api_key or os.environ["XAI_API_KEY"]
        self.voice = voice
        self.verbose = verbose

    def run(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        return asyncio.run(self.arun(record, all_schemas))

    async def arun(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        tools_schema = [_grok_tool(s) for s in tool_schemas_for(record, all_schemas)]
        system_message = SYSTEM_MESSAGE + task_type_system_suffix(record)
        audio_path = resolved_audio_path(record, duplex="full")
        frames, frame_seconds = load_duplex_stream(audio_path, SAMPLE_RATE, FRAME_MS)

        executor = MockToolExecutor()
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
                    "voice": self.voice,
                    "instructions": system_message,
                    "turn_detection": {"type": "server_vad"},
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
                        "output": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
                    },
                    **({"tools": tools_schema} if tools_schema else {}),
                },
            }))
            await self._await_session_updated(ws)

            send_task = asyncio.create_task(self._send_audio(ws, frames, frame_seconds, self.verbose))
            try:
                for step in range(1, MAX_STEPS + 1):
                    turn = await asyncio.wait_for(self._collect_response(ws, self.verbose), RESPONSE_TIMEOUT_S)
                    if turn.get("error") is not None:
                        raw_generation[f"turn{step}_error"] = turn["error"]
                        break

                    status = turn["status"]
                    function_calls = turn["function_calls"]
                    raw_generation[f"turn{step}"] = {
                        "status": status,
                        "text": turn["text"],
                        "function_calls": function_calls,
                        "response": turn["response"],
                    }
                    if status == "cancelled":
                        # server_vad cut this response off (ambient noise / disfluency
                        # pauses in the full-duplex audio); its calls were never
                        # committed and its text isn't the answer -- wait for the retry.
                        continue

                    if turn["text"]:
                        # Everything said across the whole turn, not just the last round:
                        # e.g. "Found HOTEL-954 at $330. Now checking the weather." is part
                        # of the answer the user heard (same as gemini_live's text_parts).
                        final_response = f"{final_response} {turn['text']}".strip()
                    if not function_calls:
                        break

                    for item in function_calls:
                        name = item.get("name", "")
                        try:
                            arguments = json.loads(item.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        if not isinstance(arguments, dict):
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
                print("  [grok_voice] finished streaming input audio")
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    @staticmethod
    async def _await_session_updated(ws) -> None:
        while True:
            event = json.loads(await ws.recv())
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise RuntimeError(f"Grok Voice session.update rejected: {event.get('error')}")

    @staticmethod
    async def _collect_response(ws, verbose: bool) -> Dict[str, Any]:
        """Drain events until the next response.done (or error), accumulating this
        response's function calls (from response.function_call_arguments.done) and its
        transcript/text. Anything response.done's own response.output carries is merged
        in too, deduped by call_id. Audio deltas are dropped unread."""
        calls: Dict[str, dict] = {}
        texts: List[str] = []
        while True:
            event = json.loads(await ws.recv())
            etype = event.get("type", "")
            if verbose and not etype.endswith(".delta"):
                print(f"  [grok_voice] <- {etype}")

            if etype == "error":
                return {"error": event.get("error", event)}
            if etype == "response.function_call_arguments.done":
                cid = event.get("call_id") or f"_anon{len(calls)}"
                calls[cid] = {"call_id": event.get("call_id"), "name": event.get("name", ""),
                              "arguments": event.get("arguments")}
            elif etype in TEXT_DONE_EVENTS:
                text = event.get(TEXT_DONE_EVENTS[etype])
                if text:
                    texts.append(text)
            elif etype == "response.done":
                response = event.get("response", {}) or {}
                for item in response.get("output", []) or []:
                    if item.get("type") == "function_call":
                        cid = item.get("call_id") or f"_anon{len(calls)}"
                        calls.setdefault(cid, {"call_id": item.get("call_id"), "name": item.get("name", ""),
                                               "arguments": item.get("arguments")})
                    elif item.get("type") == "message" and not texts:
                        for content in item.get("content", []) or []:
                            text = content.get("text") or content.get("transcript")
                            if text:
                                texts.append(text)
                return {
                    "status": response.get("status"),
                    "function_calls": list(calls.values()),
                    "text": "".join(texts),
                    "response": {k: v for k, v in response.items() if k != "output"},
                }
