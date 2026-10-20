# gpt-realtime

Closed-source baseline: calls OpenAI's Realtime API (`wss://api.openai.com/v1/realtime`)
directly, over the raw websocket event protocol -- no local weights, no OpenAI SDK
realtime client dependency. Fed `resolved_audio_path(record, duplex="full")` as a
real-time-paced stream of PCM16 frames (not `duplex="half"`, and not uploaded whole)
because this is a genuinely full-duplex, server-VAD-driven endpoint -- see
`models/common/duplex_audio.py`'s module docstring for why that specific input shape
is what makes the comparison against the offline/local adapters (`models/voxmind`,
`models/step_audio2`, `models/voicechat`) fair rather than just convenient.

```
export OPENAI_API_KEY=sk-...
python -m models.gpt_realtime.run_infer --limit 3 --verbose   # smoke-test first
python -m models.gpt_realtime.run_infer --output results/gpt_realtime_results.json
python evaluation/evaluate.py --src_file results/gpt_realtime_results.json
```

## Verified against a live session (2026-09-30)

Ran end to end against `wss://api.openai.com/v1/realtime` with a real `OPENAI_API_KEY`
(from `.env`) on `explicit_argument_binding_0001`: the model correctly
called `track_order(order_id="ABC123")` -- an exact match to that record's gold call --
and produced a grounded final answer citing the mock tool result. Two real issues
turned up and were fixed during that run (both already reflected in `adapter.py`):

1. **The beta wire shape is dead.** An earlier version of this adapter sent
   `OpenAI-Beta: realtime=v1` + the old flat `session.update` shape
   (`modalities`/`input_audio_format`/`turn_detection` at the top level) and got a
   live `invalid_request_error.beta_api_shape_disabled` close (code 4000) back. Fixed
   by switching to the GA shape confirmed against
   https://developers.openai.com/api/reference/resources/realtime/client-events :
   `session.type: "realtime"`, `output_modalities`, and audio config nested under
   `session.audio.input` (format is now an object, `{"type": "audio/pcm", "rate": 24000}`,
   not the bare string `"pcm16"`).
2. **A real full-duplex phenomenon, not a bug, needed handling.** This benchmark's
   `full_duplex_audio_paths` wavs are not digitally silent during the "silence" tail --
   they carry a genuine ambient noise floor (confirmed by inspecting the raw samples),
   and the disfluent transcript itself has mid-sentence pauses. Both are enough to
   retrigger `server_vad`'s `speech_started` while a response is still being generated,
   which OpenAI responds to by cancelling that response (`status: "cancelled",
   status_details.reason: "turn_detected"`) -- observed live, more than once, on this
   exact record, including one cancelled response that had already started speaking
   ("Hi there! How can I help"). The adapter now keeps waiting for the next
   `response.done` instead of treating an empty cancelled response as the record's
   final answer -- `server_vad`'s own `create_response` default means a real response
   still follows once activity quiets back down, which is exactly what happened (turn 5
   made the tool call, turn 6 gave the grounded final answer).

The specific multi-round shape exercised (several cancelled rounds -> a function_call
round -> a completed text round) is now confirmed correct end to end. What's still
unverified: parallel/multiple function calls in a single round, and any session-level
error path this one successful run didn't happen to trigger. Re-run
`--limit 1 --verbose` if OpenAI revises the GA shape again, and read the printed event
stream before trusting a full dataset run.

## Design notes

- `output_modalities=["text"]` -- same choice `models/voxmind` (`response_mode="text"`) and
  `models/voicechat` (`decode_audio=False`) already made: `evaluation/scorer.py` never
  scores audio, so there's no reason to pay for speech synthesis on the output side.
  The input side stays full-duplex real audio regardless -- that half can't be
  simplified the same way (see `models/common/duplex_audio.py`).
- `turn_detection: server_vad` is left on and never manually committed/triggered for
  turn 1 -- the trailing silence `data_synthetic/tts.py` already bakes into
  `full_duplex_audio_paths` is what the server's own VAD reacts to, the same way it
  would react to a live caller falling silent.
- Multi-step tool calls: a `response.done` may contain one or more `function_call`
  items; each gets a `function_call_output` fed back and a `response.create` is sent
  to continue, until a round comes back with no function calls left (MAX_STEPS=8 is a
  safety cap only). Every `call_id` the server names gets answered, even a repeat of an
  already-made call -- leaving one unanswered isn't something this adapter wants to
  gamble is safe under the beta protocol. A repeat reuses the first call's cached
  result instead of re-invoking `MockToolExecutor` (which has side effects, e.g.
  `add_to_cart` appending a second line), so it doesn't get double-counted or
  double-applied; only the first occurrence of each distinct `(name, args)` pair is
  counted in `actual_tool_calls`, matching every other adapter's dedup-for-scoring
  convention.
