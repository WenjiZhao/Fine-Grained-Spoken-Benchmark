"""Gemini Live API adapter: dataset record -> normalized result dict.

Uses the `google-genai` SDK's `client.aio.live.connect(...)` the same way
Full-Duplex-Bench/v1_v1.5/model_inference/gemini/inference_gemini25_native.py
and its inference_gemini31_live.py sibling already do in this repo (real
scripts, already exercised against live Gemini Live sessions for
Full-Duplex-Bench's own turn-taking runs) -- this adapter reuses that same
send-frames-at-real-time-pace connection pattern, just pointed at
`resolved_audio_path(record, duplex="full")` instead of Full-Duplex-Bench's
own input.wav, and with `tools` wired in for function calling, which neither
of those two reference scripts needed.

Output modality: `response_modalities=["AUDIO"]` plus
`output_audio_transcription={}`, NOT the bare `["TEXT"]` models/voxmind and
models/voicechat's "skip speech synthesis" choice would suggest. That was the
first thing tried here, and a live session rejected it outright as soon as
`tools` was set -- "The requested combination of response modalities (TEXT)
is not supported by the model" -- confirmed against a real gemini-3.8-live
session, not assumed. `output_audio_transcription` gets this adapter the same
end result (a text final_response, no audio bytes ever kept or decoded) by
reading `resp.server_content.output_transcription.text` instead of a direct
text part.

Turn-taking mechanics, matched to this benchmark's shape:
  - automatic_activity_detection is left at its default (enabled); the
    trailing silence in full_duplex_audio_paths (see data_synthetic/tts.py)
    is what the server's own VAD reacts to to end the user's turn, same as
    models/gpt_realtime's server_vad.
  - Unlike gpt-realtime's request/response rounds, Gemini Live's function
    calling happens *inside* a single ongoing turn: a `resp.tool_call`
    message pauses generation until every one of its `function_calls` gets
    an answer via `send_tool_response`, then generation resumes in the same
    turn without a separate "create response" call. The loop below still
    counts rounds (one round = one resp.tool_call message, which may itself
    batch several function_calls) up to MAX_STEPS as a safety cap, mirroring
    every other adapter's own MAX_STEPS convention, since nothing here
    guarantees a buggy/adversarial session couldn't otherwise loop forever.
  - Every function_call in a resp.tool_call batch gets answered by its own
    `id` (never silently dropped) -- the Live API's contract is that a
    pending tool_call blocks the session until every id in that batch is
    answered, so skipping one would hang the record instead of failing it
    cleanly. A repeat of an already-made (name, args) pair reuses the first
    call's cached MockToolExecutor result rather than re-invoking it (a
    DB-backed tool has side effects, e.g. add_to_cart appending a second
    cart line), and only the first occurrence is counted in
    actual_tool_calls -- same dedup-for-scoring convention as every other
    adapter here.

VERIFIED (2026-09-30) end to end against a live gemini-3.8-live session (real
GEMINI_API_KEY, see README.md for the full account): on a real
explicit_argument_binding record, input_audio_transcription correctly showed
the model understood the disfluent, noisy-full-duplex spoken request ("...The
order ID is ABC123.") and it called track_order(order_id="ABC123") -- an
exact match to that record's gold call, confirming resp.tool_call /
types.FunctionResponse / session.send_tool_response's shape is right. Getting
there took two live-session-driven fixes already reflected above/below: the
DEFAULT_MODEL change (several other Live models either don't exist yet, or
are "native-audio" models that reject TEXT output outright) and the
response_modalities=["AUDIO"] + output_audio_transcription switch.

FOUND AND FIXED (2026-09-30) -- root cause of an earlier "empty final_response
after any tool call" finding from a 49-record batch (5 records x 10
task_types; the ORIGINAL close_source/result/gemini_live_results.json that
batch produced showed 38/38 tool-calling records with final_response=="" and
every chained-call task_type -- resolvable_implicit_argument,
sequential_tool_call -- losing its second, dependent tool call entirely).
Two plausible client-side fixes were tried first and neither moved the
number: FunctionResponse.scheduling=WHEN_IDLE (still left set below, it's
still the documented-correct value), and a standalone probe with an explicit
"always tell the user the result out loud" system instruction. The actual
cause, per https://ai.google.dev/gemini-api/docs/live-tools : a
FunctionDeclaration's own `behavior` field -- SEPARATE from
FunctionResponse.scheduling -- defaults to Behavior.UNSPECIFIED when omitted,
and the docs' claim that omitting it behaves as BLOCKING ("Function calling
executes sequentially by default") did not hold for a live gemini-3.8-live
session: leaving `behavior` unset made generation stop dead after the FIRST
send_tool_response, every time. Setting `behavior="BLOCKING"` explicitly on
every FunctionDeclaration (see _function_declaration below) fixed both
symptoms in the same single-record probe that first isolated it: a second,
dependent tool call (get_weather, using the first call's result) fired
correctly, immediately followed by real spoken output with real
output_transcription text. The 49-record batch was re-run with this fix in
place; see README.md for the refreshed numbers.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from models.common.dataset import resolved_audio_path, task_type_system_suffix, tool_schemas_for  # noqa: E402
from models.common.duplex_audio import load_duplex_stream  # noqa: E402
from models.common.result_schema import build_result_record, normalize_calls  # noqa: E402
from tools.executor import MockToolExecutor  # noqa: E402

DEFAULT_MODEL = "gemini-3.8-live"  # NOT a "native-audio" model -- see module docstring
SEND_SAMPLE_RATE = 16000  # Live API's own native-audio input rate (see the vendored
                          # Full-Duplex-Bench gemini scripts' SEND_SAMPLE_RATE)
FRAME_MS = 32

# Upper bound on resp.tool_call rounds per record (mirrors MAX_STEPS in every other
# adapter here) -- a safety cap only, the loop normally stops as soon as the session
# reports turn_complete/generation_complete with no further tool_call.
MAX_STEPS = 8

# How long to wait for the session to produce anything further (a resp.tool_call, or
# server_content with turn_complete) after the input stream has ended / a tool response
# was just sent, before giving up on the record instead of hanging a whole run.
IDLE_TIMEOUT_S = 60.0

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


def _function_declaration(schema: dict) -> dict:
    """tools/tool_schemas.json's shape -> one entry of a Live API `tools` config's
    `function_declarations` list -- {"name", "description", "parameters", "behavior"},
    no wrapper (the wrapper is the `{"function_declarations": [...]}` dict one level
    up, built in GeminiLiveAdapter.run below).

    behavior="BLOCKING" is the one field here that is NOT optional in practice, despite
    https://ai.google.dev/gemini-api/docs/live-tools describing BLOCKING as the
    implicit default when a declaration omits it ("Function calling executes
    sequentially by default"). Confirmed live against gemini-3.8-live: leaving
    `behavior` unset (Behavior.UNSPECIFIED) makes the session go completely silent
    after the FIRST send_tool_response -- no further speech, no second tool call, even
    with FunctionResponse.scheduling=WHEN_IDLE -- across a 49-record batch (0/38
    tool-calling records got a final_response; every chained-call task_type lost its
    second call). Setting behavior="BLOCKING" explicitly on every declaration fixes
    both symptoms in the same single-record probe that first surfaced them: the model
    correctly chains a second, dependent tool call and then produces a real spoken/
    transcribed answer. See adapter.py's module docstring and README.md's history of
    this finding for the two things that were tried and did NOT fix it (scheduling,
    an explicit "confirm verbally" system instruction) before this was found."""
    return {
        "name": schema["name"],
        "description": schema.get("description", schema["name"]),
        "parameters": _with_array_items(schema.get("parameters", {"type": "object", "properties": {}, "required": []})),
        "behavior": "BLOCKING",
    }


def _with_array_items(node):
    """Copy of a JSON schema with `items: {"type": "string"}` filled in on any array
    that omits it -- the Live API rejects the whole session setup (close 1007,
    "...properties[vendor].items: missing field") for e.g. items.search's untyped
    vendor/product arrays, which OpenAI accepts as-is."""
    if isinstance(node, list):
        return [_with_array_items(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: _with_array_items(v) for k, v in node.items()}
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {"type": "string"}
    return out


def _call_key(name: str, args: dict) -> Tuple[str, str]:
    return name, json.dumps(args or {}, sort_keys=True)


class GeminiLiveAdapter:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None, verbose: bool = False):
        self.model = model
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.verbose = verbose

    def run(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        """Sync entry point for run_infer.py's sequential per-record loop. A batch
        script that wants several records running concurrently should await arun()
        directly instead (asyncio.run() can't nest inside an already-running loop)."""
        return asyncio.run(self.arun(record, all_schemas))

    async def arun(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        tool_schemas = tool_schemas_for(record, all_schemas)
        system_instruction = SYSTEM_MESSAGE + task_type_system_suffix(record)
        audio_path = resolved_audio_path(record, duplex="full")
        frames, frame_seconds = load_duplex_stream(audio_path, SEND_SAMPLE_RATE, FRAME_MS)

        config: Dict[str, Any] = {
            # response_modalities=["TEXT"] alone would match models/voxmind's and
            # models/voicechat's "skip speech synthesis, evaluation/scorer.py never
            # scores audio" choice exactly -- but a live gemini-3.8-live session
            # rejects that combination outright as soon as `tools` is set ("The
            # requested combination of response modalities (TEXT) is not supported by
            # the model", confirmed against a real session): tool-calling on this
            # model requires AUDIO output. output_audio_transcription (an empty
            # config object opts in, per the Live API) gets the text transcript of
            # that spoken output without this adapter ever needing to keep or decode
            # the audio bytes themselves -- same end result as the other adapters'
            # skip-audio choice, just reached through a transcript instead of a
            # direct text modality.
            "response_modalities": ["AUDIO"],
            "output_audio_transcription": {},
            "system_instruction": system_instruction,
        }
        if tool_schemas:
            config["tools"] = [{"function_declarations": [_function_declaration(s) for s in tool_schemas]}]

        executor = MockToolExecutor()  # fresh per record, per executor.py's own contract
        raw_calls: List[Dict[str, Any]] = []
        seen_calls = set()
        call_result_cache: Dict[Tuple[str, str], dict] = {}
        raw_generation: Dict[str, Any] = {}
        final_response = ""

        client = genai.Client(vertexai=False, api_key=self.api_key)
        async with client.aio.live.connect(model=self.model, config=config) as sess:
            send_task = asyncio.create_task(self._send_audio(sess, frames, frame_seconds, self.verbose))
            try:
                round_idx = 0
                text_parts: List[str] = []
                async for resp in self._iter_with_timeout(sess, IDLE_TIMEOUT_S, self.verbose):
                    tool_call = getattr(resp, "tool_call", None)
                    if tool_call and tool_call.function_calls:
                        round_idx += 1
                        batch_calls = []
                        responses = []
                        for fc in tool_call.function_calls:
                            name = fc.name or ""
                            args = dict(fc.args or {})
                            key = _call_key(name, args)
                            if key not in call_result_cache:
                                call_result_cache[key] = executor.call(name, **args)
                            if key not in seen_calls:
                                seen_calls.add(key)
                                raw_calls.append({"name": name, "arguments": args})
                            batch_calls.append({"name": name, "args": args})
                            responses.append(types.FunctionResponse(
                                id=fc.id, name=name, response={"result": call_result_cache[key]},
                                # Confirmed live: omitting `scheduling` (leaving it at
                                # its SCHEDULING_UNSPECIFIED default) delivers the
                                # result into context WITHOUT ever prompting the model
                                # to continue speaking -- 38/39 tool-calling records in
                                # a 49-record smoke batch came back with a correct
                                # actual_tool_calls entry but final_response="" (every
                                # record with NO tool call got a real final_response;
                                # every one WITH a tool call didn't, bar one). WHEN_IDLE
                                # is the Live API's explicit "prompt a response once
                                # nothing else is pending" scheduling value -- see
                                # google.genai.types.FunctionResponseScheduling.
                                scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
                            ))
                        raw_generation[f"turn{round_idx}_tool_call"] = batch_calls
                        await sess.send_tool_response(function_responses=responses)
                        if round_idx >= MAX_STEPS:
                            break
                        continue

                    sc = getattr(resp, "server_content", None)
                    transcription = sc and getattr(sc, "output_transcription", None)
                    if transcription and getattr(transcription, "text", None):
                        text_parts.append(transcription.text)
                    if sc and (getattr(sc, "turn_complete", False) or getattr(sc, "generation_complete", False)):
                        final_response = "".join(text_parts)
                        raw_generation[f"turn{round_idx + 1}_final"] = final_response
                        break
            finally:
                send_task.cancel()

        return build_result_record(
            record, normalize_calls(raw_calls), executor.tool_results(), final_response, raw_generation
        )

    @staticmethod
    async def _send_audio(sess, frames: List[bytes], frame_seconds: float, verbose: bool) -> None:
        try:
            for frame in frames:
                await sess.send_realtime_input(audio={"data": frame, "mime_type": "audio/pcm"})
                await asyncio.sleep(frame_seconds)
            await sess.send_realtime_input(audio_stream_end=True)
            if verbose:
                print("  [gemini_live] finished streaming input audio")
        except asyncio.CancelledError:
            pass

    @staticmethod
    async def _iter_with_timeout(sess, timeout_s: float, verbose: bool):
        """sess.receive()'s own async iterator, wrapped so it raises TimeoutError
        instead of hanging forever if the session goes idle with nothing further to
        say. The generator itself must be created exactly once (calling
        sess.receive() again on every step would each time start a brand new iterator
        rather than resume the same one)."""
        gen = sess.receive()
        while True:
            try:
                resp = await asyncio.wait_for(gen.__anext__(), timeout_s)
            except StopAsyncIteration:
                return
            if verbose:
                kind = "tool_call" if getattr(resp, "tool_call", None) else "server_content"
                print(f"  [gemini_live] <- {kind}")
            yield resp
