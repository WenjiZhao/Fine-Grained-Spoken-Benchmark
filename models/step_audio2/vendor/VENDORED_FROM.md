# Vendored source

This directory is a vendored, trimmed snapshot of stepfun-ai/Step-Audio2's
base inference code -- plain files, not a git submodule, and not the full
repo.

- Upstream: https://github.com/stepfun-ai/Step-Audio2.git
- Commit: 472b6274ffa705fa2c101968187683ec46920d57
- Vendored on: 2026-08-31

Only `stepaudio2.py` (the `StepAudio2Base`/`StepAudio2` classes -- the local
`transformers` path, not the vLLM HTTP client in upstream's
`stepaudio2vllm.py`) and `utils.py` (audio loading / mel spectrogram
helpers) are included. The upstream repo's TTS stack (`token2wav.py`,
`flashcosyvoice/`) is intentionally left out: `models/step_audio2/adapter.py`
only decodes text (no `<tts_start>` in any message), so no speech synthesis
dependency is needed -- same reasoning as `models/voxmind/vendor/`.

One deliberate change from upstream: `StepAudio2Base.__init__`/`StepAudio2.__init__`
take `device`/`torch_dtype` parameters instead of upstream's hardcoded
`.cuda()` / `torch.bfloat16`, so `--device` on the command line actually
does something. This is the same change `models/voxmind/vendor/runtime/model.py`
already made to this exact same base class (VoxMind is a further fine-tune
of Step-Audio2, so the two vendor copies are close cousins -- compare
`models/voxmind/vendor/VENDORED_FROM.md`). No other logic was touched.

Model weights are never vendored -- point `--checkpoint` at your own copy
(e.g. a local `stepfun-ai/Step-Audio-2-mini` download).

For updates, pull from upstream directly rather than this repo.
