# Vendored source

This directory is a vendored, trimmed snapshot of MM-Speech/VoxMind's
inference code — plain files, not a git submodule, and not the full repo.

- Upstream: https://github.com/MM-Speech/VoxMind.git
- Commit: 115761b
- Vendored on: 2026-08-25

Only `runtime/` (the `VoxMind`/`StepAudio2`/`StepAudio2Base` classes,
system-prompt builder, `<tool_call>` parser) and `utils.py` (audio loading /
mel spectrogram helpers) are included. The upstream repo's TTS stack
(`token2wav.py`, `cosyvoice2/`, `flashcosyvoice/`) is intentionally left out:
`models/voxmind/adapter.py` only runs the model in `response_mode="text"`
for tool-calling evaluation, so no speech synthesis dependency is needed.
Model weights are never vendored — point `--checkpoint` at your own copy.

For updates, pull from upstream directly rather than this repo.
