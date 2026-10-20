# Step-Audio2

Local `transformers` path, chosen over standing up the official vLLM Docker
image (`stepfun2025/vllm:step-audio-2-v20250909`, which bundles a patched
vLLM with `--tool-call-parser step_audio_2` -- real infra to stand up on
this cluster, and gets you a *verified* structured `tool_calls` parser at
the cost of a whole extra service). This adapter instead runs the
un-fine-tuned `Step-Audio-2-mini` checkpoint in-process, same shape as
`models/voxmind/adapter.py` (VoxMind is itself Step-Audio2 plus a further
fine-tune).

```
python -m models.step_audio2.run_infer --checkpoint /path/to/Step-Audio-2-mini \
    [--dataset data/] [--output results/step_audio2_results.json] \
    [--limit 10] [--device cuda] [--verbose]
```

**CAVEAT -- unverified tool-call text format.** The upstream repo's own demo
(`examples.py::tool_call_test`) never parses the model's tool-call text at
all -- it hard-codes a canned search result and continues regardless of
what the model said, so there's no documented sample of how this
un-fine-tuned checkpoint spells "call tool X with args Y" in its own
output. `adapter.py`'s `SYSTEM_MESSAGE` explicitly *teaches* the model the
same `<tool_call>[{"name":..., "arguments":{...}}]</tool_call>` format
VoxMind's own fine-tuned prompt teaches, and `_extract_tool_calls()` has a
bare-JSON fallback scan in case the model ignores that and emits something
else JSON-shaped. **Run `--limit 3 --verbose` first** and read the printed
turn-1 raw text before trusting a full run -- if neither the taught tag nor
the fallback ever fires, that's a sign the real format needs to be added
based on what the checkpoint actually outputs.

See `adapter.py`'s module docstring and `vendor/VENDORED_FROM.md` for the
full reasoning and provenance.
