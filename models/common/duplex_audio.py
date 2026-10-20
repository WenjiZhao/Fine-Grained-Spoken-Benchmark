"""Full-duplex audio streaming helpers shared by the closed-source realtime-API
adapters (models/gpt_realtime, models/gemini_live).

Both providers are genuinely full-duplex, server-VAD-driven APIs: the model
decides *when* to start responding by watching the arrival of audio frames,
not by waiting for an explicit "end of turn" signal from the client. That is
exactly why these two adapters -- unlike the offline/local adapters
(models/voxmind, models/step_audio2, models/voicechat), which hand a
model.generate() call the whole clip at once and get one batch response back
-- must use resolved_audio_path(record, duplex="full") instead of the "half"
default, and must feed it in as a real-time-paced stream of small frames
rather than one big upload. data_synthetic/tts.py's full_duplex_audio_paths
already append response_silence_seconds (20s by default) of exact
zero-amplitude trailing frames after the spoken request ends specifically so
a server-VAD endpoint has silence to react to and end the user's turn on its
own, the same way it would with a live caller who stops talking and waits.

Pacing (asyncio.sleep between frames, not just correct frame content) matters
for the same reason: sending all frames back-to-back as fast as possible
would make the provider's VAD see the whole clip arrive in a few
milliseconds, which is nothing like the real-time turn-taking behavior this
benchmark is trying to measure. This mirrors the timing discipline already
used by Full-Duplex-Bench/v1_v1.5/model_inference/gemini/
inference_gemini25_native.py's sender() coroutine, ported here as a
provider-agnostic helper so models/gpt_realtime/adapter.py and
models/gemini_live/adapter.py don't each reimplement wav-loading/resampling/
chunking themselves.
"""

import wave
from math import gcd
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.signal import resample_poly


def _pcm_to_int16(raw: bytes, sampwidth: int, path: Path) -> np.ndarray:
    """Integer-PCM bytes at any of the bit depths WAV actually uses -> 1-D int16,
    by truncating each sample to its top 16 bits (standard bit-depth reduction --
    matches what any int8/int16/int24/int32 PCM decoder does when asked for less
    precision than it's given). data_synthetic/tts.py's own writes are 16-bit, but
    data_synthetic/final_data's full_duplex/half_duplex wavs turned out to be 32-bit
    PCM at 48kHz (confirmed via `file`/hex-dump against a real record, not assumed) --
    handling the full width range here means this adapter isn't silently wrong the
    next time a wav's provenance changes bit depth again."""
    if sampwidth == 2:
        return np.frombuffer(raw, dtype=np.int16)
    if sampwidth == 4:
        return (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    if sampwidth == 1:
        # WAV's 8-bit PCM is unsigned with a 128 midpoint, unlike every wider width.
        return (((np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128) << 8)).astype(np.int16)
    if sampwidth == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int32 = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int8).astype(np.int32) << 16)  # top byte carries the sign
        )
        return (as_int32 >> 8).astype(np.int16)
    raise ValueError(f"{path}: unsupported PCM sample width ({sampwidth * 8}-bit)")


def load_pcm16_mono(path: Path) -> Tuple[np.ndarray, int]:
    """(samples, sample_rate) -- samples is 1-D int16, downmixed to mono if the wav
    file is multi-channel (data_synthetic/tts.py's own wavs are already mono, but this
    holds for any full_duplex_audio_paths entry regardless of provenance)."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    data = _pcm_to_int16(raw, sampwidth, path)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    return data, sample_rate


def resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample -- exact for the rational ratios actually in play here
    (tts.py's ElevenLabs output is 44100Hz; OpenAI Realtime's pcm16 is fixed at
    24000Hz both directions; Gemini Live's native-audio models take 16000Hz in /
    emit 24000Hz out)."""
    if src_rate == dst_rate:
        return samples
    g = gcd(src_rate, dst_rate)
    resampled = resample_poly(samples.astype(np.float64), dst_rate // g, src_rate // g)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def chunk_frames(samples: np.ndarray, frame_samples: int) -> List[bytes]:
    """Zero-pad to a whole number of frame_samples-sample frames; return each frame's
    raw PCM16 little-endian bytes, in order."""
    pad = (-len(samples)) % frame_samples
    if pad:
        samples = np.pad(samples, (0, pad))
    return [samples[i : i + frame_samples].tobytes() for i in range(0, len(samples), frame_samples)]


def load_duplex_stream(path: Path, target_rate: int, frame_ms: int = 32) -> Tuple[List[bytes], float]:
    """wav file -> mono PCM16 @ target_rate -> frame_ms-long frames.

    Returns (frames, frame_seconds); frame_seconds is how long the caller should
    asyncio.sleep() between sends to keep real-time pacing (see this module's
    docstring for why pacing, not just frame content, is load-bearing here).
    """
    samples, src_rate = load_pcm16_mono(path)
    samples = resample_int16(samples, src_rate, target_rate)
    frame_samples = int(round(target_rate * frame_ms / 1000))
    frames = chunk_frames(samples, frame_samples)
    return frames, frame_samples / target_rate
