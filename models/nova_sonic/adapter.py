"""Amazon Nova 2.5 Sonic adapter (Bedrock InvokeModelWithBidirectionalStream):
dataset record -> normalized result dict.

Same benchmark shape as models/gpt_realtime and models/grok_voice: the record's
full_duplex audio (resolved_audio_path(record, duplex="full")) is streamed as
real-time-paced PCM16 frames (models/common/duplex_audio.py), the model's own
turn detection decides when the user is done, tool calls are answered from
MockToolExecutor until the model gives a spoken final answer, and the final
answer's text is read from Nova's FINAL-stage ASSISTANT transcript (output
audio is generated -- Nova has no text-only output mode -- but never kept).

Transport / auth:
  - Bedrock's bidirectional stream is only exposed through the new
    smithy-based `aws_sdk_bedrock_runtime` SDK (Python >= 3.12, so this model
    has its own venv -- see README.md), over HTTP/2 via awscrt.
  - The key in .env (repo root) is a Bedrock API key ("ABSK..."), not an
    IAM access key pair. The SDK only ships SigV4, so _BearerAuthScheme below
    plugs in `Authorization: Bearer <key>` instead, with no per-event signer
    (event_signer() -> None). If $AWS_ACCESS_KEY_ID/$AWS_SECRET_ACCESS_KEY are
    passed instead, the SDK's normal SigV4 path is used.

Event protocol (https://docs.aws.amazon.com/nova/latest/nova2-userguide/
sonic-input-events.html / sonic-output-events.html):
  in:  sessionStart -> promptStart(tools) -> SYSTEM text content ->
       USER AUDIO content (audioInput frames, left open for the whole record)
       ... per toolUse: TOOL content (contentStart/toolResult/contentEnd) ...
       -> contentEnd(audio) -> promptEnd -> sessionEnd
  out: per assistant turn: USER ASR text, optional toolUse, ASSISTANT
       SPECULATIVE text, AUDIO, ASSISTANT FINAL text; a contentEnd with
       stopReason INTERRUPTED means barge-in (the audio's noise floor /
       disfluency pauses can trigger it, same as gpt_realtime's cancelled
       responses) and that text is dropped.

Turn-taking: the record ends once an assistant turn is fully spoken (see the
final_parts comment in arun) -- the same "first completed text-only
turn is the answer" rule as gpt_realtime/grok_voice. Repeated (name, args)
calls reuse the cached executor result and are counted once.
"""

import asyncio
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from smithy_core.shapes import ShapeID
from smithy_http import Field
from smithy_http.aio.crt import AWSCRTHTTPClient

BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from models.common.dataset import resolved_audio_path, task_type_system_suffix, tool_schemas_for  # noqa: E402
from models.common.duplex_audio import load_duplex_stream  # noqa: E402
from models.common.result_schema import build_result_record, normalize_calls  # noqa: E402
from tools.executor import MockToolExecutor  # noqa: E402

DEFAULT_MODEL = "amazon.nova-2-5-sonic"  # Nova 2.5 Sonic; Nova 2 Sonic is "amazon.nova-2-sonic-v1:0"
DEFAULT_REGION = "us-east-1"
DEFAULT_VOICE = "matthew"
SAMPLE_RATE = int(os.environ.get("NOVA_INPUT_RATE", 16000))
OUTPUT_SAMPLE_RATE = 24000
FRAME_MS = 32

MAX_STEPS = 8
RESPONSE_TIMEOUT_S = 60.0
FINAL_WAIT_S = 15.0  # max wait for playback-paced FINAL transcripts after the audio ends
GRACE_S = 3.0  # wait for a follow-up answer once the current one is fully transcribed

# Same system prompt as the other closed-source adapters so the comparison is fair.
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


class _BearerSigner:
    def __init__(self, api_key: str):
        self._api_key = api_key

    async def sign(self, *, request, identity, properties):
        request.fields.set_field(Field(name="Authorization", values=[f"Bearer {self._api_key}"]))
        return request


class _NoIdentityResolver:
    async def get_identity(self, *, properties):
        from smithy_core.interfaces.identity import Identity

        class _Anon(Identity):
            expiration = None

        return _Anon()


class _BearerAuthScheme:
    """Bedrock API key auth: one `Authorization: Bearer` header on the HTTP/2
    request, events sent unsigned."""

    scheme_id = ShapeID("smithy.api#httpBearerAuth")

    def __init__(self, api_key: str):
        self._signer = _BearerSigner(api_key)

    def identity_properties(self, *, context):
        return {}

    def identity_resolver(self, *, context):
        return _NoIdentityResolver()

    def signer_properties(self, *, context):
        return {}

    def signer(self):
        return self._signer

    def event_signer(self, *, request):
        return None


class _BearerAuthResolver:
    def resolve_auth_scheme(self, *, auth_parameters):
        from smithy_core.auth import AuthOption

        return [AuthOption(scheme_id=_BearerAuthScheme.scheme_id, identity_properties={}, signer_properties={})]


def _nova_tool(schema: dict) -> dict:
    return {"toolSpec": {
        "name": schema["name"],
        "description": schema.get("description", schema["name"]),
        "inputSchema": {"json": json.dumps(
            schema.get("parameters", {"type": "object", "properties": {}, "required": []}))},
    }}


def _call_key(name: str, arguments: dict) -> Tuple[str, str]:
    return name, json.dumps(arguments or {}, sort_keys=True)


def _generation_stage(content_start: dict) -> Optional[str]:
    try:
        return json.loads(content_start.get("additionalModelFields") or "{}").get("generationStage")
    except (json.JSONDecodeError, AttributeError):
        return None


class NovaSonicAdapter:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 region: str = DEFAULT_REGION, voice: str = DEFAULT_VOICE, verbose: bool = False):
        self.model = model
        self.api_key = api_key
        self.region = region
        self.voice = voice
        self.verbose = verbose

    async def _client(self) -> AsyncBedrockRuntimeClient:
        kwargs = dict(endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
                      region=self.region, transport=AWSCRTHTTPClient())
        if self.api_key:
            kwargs.update(auth_scheme_resolver=_BearerAuthResolver(),
                          auth_schemes={_BearerAuthScheme.scheme_id: _BearerAuthScheme(self.api_key)})
        # else: the SDK's default AWS credential chain (env vars / profile) + SigV4.
        return AsyncBedrockRuntimeClient(config=await AsyncBedrockRuntimeConfig.resolve(**kwargs))

    def run(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        return asyncio.run(self.arun(record, all_schemas))

    async def arun(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        tools = [_nova_tool(s) for s in tool_schemas_for(record, all_schemas)]
        system_message = SYSTEM_MESSAGE + task_type_system_suffix(record)
        frames, frame_seconds = load_duplex_stream(resolved_audio_path(record, duplex="full"), SAMPLE_RATE, FRAME_MS)

        executor = MockToolExecutor()
        raw_calls: List[Dict[str, Any]] = []
        seen_calls = set()
        call_result_cache: Dict[Tuple[str, str], dict] = {}
        raw_generation: Dict[str, Any] = {"events": []}
        final_response = ""

        prompt_name = str(uuid.uuid4())
        audio_content = str(uuid.uuid4())
        stream = await (await self._client()).invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model))
        send_lock = asyncio.Lock()

        async def send(event: dict) -> None:
            async with send_lock:
                await stream.input_stream.send(InvokeModelWithBidirectionalStreamInputChunk(
                    value=BidirectionalInputPayloadPart(bytes_=json.dumps({"event": event}).encode("utf-8"))))

        async def send_text(role: str, text: str) -> None:
            name = str(uuid.uuid4())
            await send({"contentStart": {"promptName": prompt_name, "contentName": name, "type": "TEXT",
                                         "interactive": False, "role": role,
                                         "textInputConfiguration": {"mediaType": "text/plain"}}})
            await send({"textInput": {"promptName": prompt_name, "contentName": name, "content": text}})
            await send({"contentEnd": {"promptName": prompt_name, "contentName": name}})

        async def send_audio() -> None:
            try:
                t0 = time.monotonic()
                for i, frame in enumerate(frames, start=1):
                    await send({"audioInput": {"promptName": prompt_name, "contentName": audio_content,
                                               "content": base64.b64encode(frame).decode("ascii")}})
                    # Pace against the wall clock, not sleep-after-send: the HTTP/2 send
                    # itself takes a few ms, which otherwise accumulates into ~10% drift.
                    await asyncio.sleep(max(0.0, t0 + i * frame_seconds - time.monotonic()))
                    if self.verbose and i % 100 == 0:
                        print(f"  [nova_sonic] sent {i * frame_seconds:.2f}s audio in {time.monotonic() - t0:.2f}s")
                if self.verbose:
                    print(f"  [nova_sonic] finished streaming input audio: {time.monotonic() - t0:.2f}s wall "
                          f"for {len(frames) * frame_seconds:.2f}s of audio")
            except asyncio.CancelledError:
                pass

        await send({"sessionStart": {
            "inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7},
            "turnDetectionConfiguration": {"endpointingSensitivity": os.environ.get("NOVA_ENDPOINTING", "MEDIUM")},
        }})
        await send({"promptStart": {
            "promptName": prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {"mediaType": "audio/lpcm", "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                                         "sampleSizeBits": 16, "channelCount": 1, "voiceId": self.voice,
                                         "encoding": "base64", "audioType": "SPEECH"},
            "toolUseOutputConfiguration": {"mediaType": "application/json"},
            **({"toolConfiguration": {"tools": tools}} if tools else {}),
        }})
        await send_text("SYSTEM", system_message)
        await send({"contentStart": {"promptName": prompt_name, "contentName": audio_content, "type": "AUDIO",
                                     "interactive": True, "role": "USER",
                                     "audioInputConfiguration": {"mediaType": "audio/lpcm",
                                                                 "sampleRateHertz": SAMPLE_RATE,
                                                                 "sampleSizeBits": 16, "channelCount": 1,
                                                                 "audioType": "SPEECH", "encoding": "base64"}}})

        t_start = time.monotonic()
        send_task = asyncio.create_task(send_audio())
        _, output_stream = await stream.await_output()
        open_blocks: Dict[str, dict] = {}  # contentId -> {"role", "type", "stage", "text"}
        tool_rounds = 0
        # How Nova speaks an answer (observed live):
        #   - sentence by sentence: one SPECULATIVE text block per sentence before its
        #     audio, then one FINAL text block per sentence, paced to real-time playback
        #     (seconds after the audio's own END_TURN), and any FINAL block may carry
        #     END_TURN, not just the last one;
        #   - with parallel/chained tool calls it often answers as soon as the FIRST
        #     result is in ("...still waiting for the weather"), then ~0.5s after that
        #     answer finishes playing starts a second answer with the remaining results.
        # So: once the audio has ended, wait for the FINAL blocks to catch up with the
        # SPECULATIVE ones (up to FINAL_WAIT_S; any still missing fall back to their
        # SPECULATIVE text -- the two only differ on barge-in, which resets the turn),
        # then wait GRACE_S more for a follow-up answer. The final_response is all of it
        # joined, i.e. everything the user heard (same as the other adapters).
        final_parts: List[str] = []
        speculative_parts: List[str] = []
        audio_ended = False
        try:
            while True:
                caught_up = audio_ended and len(final_parts) >= len(speculative_parts)
                timeout = GRACE_S if caught_up else FINAL_WAIT_S if audio_ended else RESPONSE_TIMEOUT_S
                try:
                    chunk = await asyncio.wait_for(output_stream.receive(), timeout)
                except asyncio.TimeoutError:
                    if audio_ended:
                        break
                    raise
                if chunk is None:
                    break
                payload = getattr(getattr(chunk, "value", None), "bytes_", None)
                if not payload:
                    raw_generation["error"] = repr(chunk)
                    break
                event = json.loads(payload.decode("utf-8")).get("event", {})
                etype = next(iter(event), "")
                body = event.get(etype, {})
                if etype not in ("audioOutput", "usageEvent"):
                    if self.verbose:
                        detail = body.get("content") if etype in ("textOutput", "toolUse") else (
                            body.get("stopReason") or body.get("additionalModelFields") or "")
                        print(f"  [nova_sonic] <- {etype} {body.get('role', '')} {str(detail)[:120]}")
                    raw_generation["events"].append({etype: {**{k: v for k, v in body.items()
                                                                if k not in ("sessionId", "promptName", "completionId")},
                                                             "_t": round(time.monotonic() - t_start, 2)}})

                if etype == "contentStart":
                    if body.get("role") == "ASSISTANT" and _generation_stage(body) != "FINAL":
                        audio_ended = False  # a (follow-up) answer is being generated
                    open_blocks[body.get("contentId")] = {"role": body.get("role"), "type": body.get("type"),
                                                          "stage": _generation_stage(body), "text": ""}
                elif etype == "textOutput":
                    block = open_blocks.get(body.get("contentId"))
                    if block is not None:
                        block["text"] += body.get("content", "")
                elif etype == "toolUse":
                    tool_rounds += 1
                    if tool_rounds > MAX_STEPS:
                        raw_generation["error"] = f"exceeded MAX_STEPS={MAX_STEPS} tool calls"
                        break
                    name = body.get("toolName", "")
                    try:
                        arguments = json.loads(body.get("content") or "{}")
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
                    result_content = str(uuid.uuid4())
                    await send({"contentStart": {"promptName": prompt_name, "contentName": result_content,
                                                 "interactive": False, "type": "TOOL", "role": "TOOL",
                                                 "toolResultInputConfiguration": {
                                                     "toolUseId": body.get("toolUseId"), "type": "TEXT",
                                                     "textInputConfiguration": {"mediaType": "text/plain"}}}})
                    await send({"toolResult": {"promptName": prompt_name, "contentName": result_content,
                                               "content": json.dumps(call_result_cache[key], ensure_ascii=False)}})
                    await send({"contentEnd": {"promptName": prompt_name, "contentName": result_content}})
                elif etype == "contentEnd":
                    block = open_blocks.pop(body.get("contentId"), None)
                    stop = body.get("stopReason")
                    if not block:
                        continue
                    if block["role"] == "USER" or (block["role"] == "ASSISTANT" and stop == "INTERRUPTED"):
                        # New user speech, or barge-in cut the response off: the model
                        # will answer again, so start collecting a fresh answer.
                        final_parts, speculative_parts, audio_ended = [], [], False
                    elif block["role"] == "ASSISTANT" and block["type"] == "TEXT":
                        if block["stage"] == "SPECULATIVE" and block["text"].strip():
                            speculative_parts.append(block["text"].strip())
                        elif block["stage"] == "FINAL" and block["text"].strip():
                            final_parts.append(block["text"].strip())
                    elif block["role"] == "ASSISTANT" and block["type"] == "AUDIO" and stop == "END_TURN":
                        audio_ended = True
                elif etype == "completionEnd":
                    break
            final_response = " ".join(final_parts + speculative_parts[len(final_parts):])
        finally:
            send_task.cancel()
            try:
                await send({"contentEnd": {"promptName": prompt_name, "contentName": audio_content}})
                await send({"promptEnd": {"promptName": prompt_name}})
                await send({"sessionEnd": {}})
                await stream.input_stream.close()
            except Exception:
                pass

        return build_result_record(
            record, normalize_calls(raw_calls), executor.tool_results(), final_response, raw_generation
        )
