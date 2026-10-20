# gemini-live

Closed-source baseline: calls Google's Gemini Live API via the official `google-genai`
SDK (`client.aio.live.connect`) -- no local weights. Fed
`resolved_audio_path(record, duplex="full")` as a real-time-paced stream of PCM16
frames (not `duplex="half"`, and not uploaded whole) because this is a genuinely
full-duplex, server-VAD-driven endpoint -- see `models/common/duplex_audio.py`'s
module docstring for why that specific input shape is what makes the comparison
against the offline/local adapters (`models/voxmind`, `models/step_audio2`,
`models/voicechat`) fair rather than just convenient. The connection/pacing pattern
itself is a close copy of this repo's own already-working
`Full-Duplex-Bench/v1_v1.5/model_inference/gemini/inference_gemini25_native.py`, just
pointed at this benchmark's audio and with `tools` wired in for function calling.

```
export GEMINI_API_KEY=...
python -m models.gemini_live.run_infer --limit 3 --verbose   # smoke-test first
python -m models.gemini_live.run_infer --output results/gemini_live_results.json
python evaluation/evaluate.py --src_file results/gemini_live_results.json
```

## Verified against a live session (2026-09-30)

Ran end to end against a real `gemini-3.8-live` session (`GEMINI_API_KEY` from
`.env`) on `explicit_argument_binding_0001`: `input_audio_transcription`
confirmed the model correctly understood the disfluent, noisy-full-duplex spoken
request ("...The order ID is ABC123."), and it called
`track_order(order_id="ABC123")` -- an exact match to that record's gold call. Getting
a live session working at all took two real fixes, both already reflected in
`adapter.py`:

1. **Model choice.** The originally-picked `gemini-2.5-flash-native-audio-preview-*`
   and `gemini-3.1-flash-live-preview` models both refused `response_modalities:
   ["TEXT"]` outright ("The requested combination of response modalities (TEXT) is not
   supported by the model") -- they're audio-only. `gemini-2.0-flash-live-001`, the
   older non-native-audio model this repo's own
   `inference_gemini25_native.py` comments as deprecated, no longer exists at all
   (`models/gemini-2.0-flash-live-001 is not found`). `client.models.list()` against a
   real key was used to find what's actually live right now; `gemini-3.8-live` is what
   this adapter defaults to (`gemini-3.8-live-extended-thinking` also exists but
   requires a `thinking_level` this adapter doesn't set).
2. **Output modality.** Even `gemini-3.8-live` rejects `response_modalities: ["TEXT"]`
   the moment `tools` is set in the session config -- confirmed live, not assumed.
   Function calling on this model requires `["AUDIO"]`. Fixed by keeping `["AUDIO"]`
   and adding `output_audio_transcription: {}`, which gets this adapter the same
   text-only `final_response` models/voxmind's/models/voicechat's "skip audio" choice
   was going for, just read from `resp.server_content.output_transcription.text`
   instead of a direct text part -- no audio bytes are ever kept or decoded.

## Found and fixed: `FunctionDeclaration.behavior` must be set explicitly

An initial 49-record batch (5 records x 10 task_types) found every tool-calling
record coming back with `final_response == ""` (38/38) and every chained-call
task_type losing its second tool call entirely -- `resolvable_implicit_argument`
(resolver call -> target call) and `sequential_tool_call` (2 dependent calls) both
5/5 stuck on just the first call. Two plausible fixes were tried first and neither
moved the numbers: `FunctionResponse.scheduling=WHEN_IDLE` (still left set in
`adapter.py`, it's still the documented-correct value) and a standalone probe with an
explicit "always tell the user the result out loud" system instruction.

The actual cause, per https://ai.google.dev/gemini-api/docs/live-tools : a
`FunctionDeclaration` has its own `behavior` field (`UNSPECIFIED` / `BLOCKING` /
`NON_BLOCKING`), separate from `FunctionResponse.scheduling`, and it defaults to
`UNSPECIFIED` when omitted. The docs describe omitting it as behaving like `BLOCKING`
("Function calling executes sequentially by default") -- that did not hold in a live
`gemini-3.8-live` session: leaving it unset made generation stop dead after the
*first* `send_tool_response`, every time. `_function_declaration()` now sets
`"behavior": "BLOCKING"` explicitly on every tool, and a re-run of the full
49-record batch (current `close_source/result/gemini_live_results.json`) confirms it:

| | before the fix | after the fix |
|---|---|---|
| tool-calling records with `final_response == ""` | 38/38 | **0/39** |
| `resolvable_implicit_argument` (needs 2 chained calls) | 5/5 got only the resolver call | **5/5 got both calls** |
| `sequential_tool_call` (needs 2 dependent calls) | 5/5 got only the first call | **5/5 got both calls** |
| `parallel_tool_call` (needs 2 independent calls) | 2/5 got both | **3/5 got both** |

The 2 remaining `parallel_tool_call` records that still get only one call aren't the
same bug: one of them (`parallel_tool_call_0002`) explicitly answers "I don't have the
tools to check for vehicle availability" -- its second gold tool is
`vehicle_check_availability`, which `[dataset] warning: record parallel_tool_call_0002
references unknown tool 'vehicle_check_availability'` already flags as not resolving
against `tools/tool_schemas.json` (a data-curation naming mismatch, not a
gemini_live-specific problem) -- so the model correctly recognized it had no such tool
available.

## Design notes

- Every `FunctionDeclaration` sets `"behavior": "BLOCKING"` explicitly -- see the
  "Found and fixed" section above for why this is required in practice, not optional.
- Unlike `models/gpt_realtime`'s request/response rounds, Gemini Live's function
  calling happens *inside* a single ongoing turn: a `resp.tool_call` message pauses
  generation until every one of its `function_calls` is answered via
  `send_tool_response`, then generation resumes in the same turn -- no separate
  "create response" call. The adapter still counts rounds (one round = one
  `resp.tool_call` message, which may itself batch several calls) up to MAX_STEPS=8 as
  a safety cap.
- Every `function_call` in a batch gets answered by its own `id` -- the Live API's
  contract is that a pending tool_call blocks the session until every id in that batch
  is answered, so silently skipping one would hang the record instead of failing it
  cleanly. A repeat of an already-made `(name, args)` pair reuses the first call's
  cached `MockToolExecutor` result instead of re-invoking it (a DB-backed tool has side
  effects, e.g. `add_to_cart` appending a second cart line), and only the first
  occurrence is counted in `actual_tool_calls` -- same dedup-for-scoring convention as
  every other adapter here.
