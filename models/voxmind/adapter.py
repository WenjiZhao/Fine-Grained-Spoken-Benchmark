"""VoxMind adapter: dataset record -> normalized result dict.

Multi-step ReAct loop over the vendored runtime.VoxMind, extending the pattern
demonstrated in the upstream repo's agent_demo.py (run_weather_tool_voice_case):
each turn lets the model decide whether/what to call, tool calls are executed
against a fresh MockToolExecutor, results are fed back as "observation"
messages, and the loop continues until a turn emits no new tool call (that
turn's answer is the final response) or MAX_STEPS is hit. Records with no tool
call stop after turn 1.

response_mode="text" throughout -- evaluation/scorer.py never scores audio,
so we skip TTS entirely (and with it the token2wav/cosyvoice2/flashcosyvoice
dependency; see vendor/VENDORED_FROM.md).
"""

import sys
from pathlib import Path
import json
from typing import Any, Dict, List

BENCH_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from runtime import DEFAULT_SYSTEM_PROMPT, VoxMind  # noqa: E402

from models.common.dataset import available_tool_names, resolved_audio_path, tool_schemas_for  # noqa: E402
from models.common.result_schema import build_result_record, normalize_calls  # noqa: E402
from tools.executor import MockToolExecutor  # noqa: E402

GENERATION_KWARGS = dict(
    max_new_tokens=2048,
    temperature=0.6,
    repetition_penalty=1.05,
    top_p=0.9,
    do_sample=True,
)
POST_THINK_PREFIX = "After careful reasoning, here is my detailed answer:\n"

# Upper bound on generate() rounds per record (each round may emit tool calls
# whose results are fed back before the next round). Safety cap only -- the
# loop normally stops as soon as a round emits no new tool call.
MAX_STEPS = 8


class VoxMindAdapter:
    def __init__(self, checkpoint: str, device: str = "cuda", extra_context: dict | None = None,
                 response_mode: str = "text"):
        self.model = VoxMind(checkpoint, device=device)
        self.extra_context = extra_context or {}
        # "speech" swaps the forced post-think prefix for <tts_start> (upstream
        # agent_demo.py's default); only the decoded text is scored, audio tokens
        # are dropped. Audio tokens share max_new_tokens, so match upstream's 4096.
        self.response_mode = response_mode
        self.generation_kwargs = dict(GENERATION_KWARGS)
        if response_mode == "speech":
            self.generation_kwargs["max_new_tokens"] = 4096

    def run(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        tools = tool_schemas_for(record, all_schemas)
        system_prompt = self.model.build_system_prompt(
            DEFAULT_SYSTEM_PROMPT, tools, extra_context=self.extra_context
        )
        audio_path = resolved_audio_path(record)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "audio", "audio": str(audio_path)}]},
        ]

        executor = MockToolExecutor(allowed_tools=available_tool_names(record))  # fresh per record, per executor.py's own contract
        raw_calls: List[Dict[str, Any]] = []
        seen_calls = set()
        raw_generation: Dict[str, Any] = {}
        final_response = ""

        # Keep feeding tool results back while the model keeps emitting new
        # calls, so sequential tasks (call B needs call A's result) can reach
        # call B. Stops on a turn with no tool call, on a repeat of an
        # already-made call (don't retry the same request), or at MAX_STEPS.
        for step in range(1, MAX_STEPS + 1):
            response = self.model.generate(
                messages, response_mode=self.response_mode, post_think_prefix=POST_THINK_PREFIX, **self.generation_kwargs
            )
            raw_generation[f"turn{step}"] = self._turn_debug(response)
            final_response = response.answer

            new_calls = []
            for call in self.model.parse_tool_calls(response.answer):
                key = (call.get("name", ""), json.dumps(call.get("arguments") or {}, sort_keys=True))
                if key not in seen_calls:
                    seen_calls.add(key)
                    new_calls.append(call)
            if not new_calls:
                break

            n_before = len(executor.tool_results())
            for call in new_calls:
                executor.call(call.get("name", ""), **(call.get("arguments") or {}))
            raw_calls.extend(new_calls)

            messages.append({"role": "assistant", "content": f"{self.model.THINK_START}\n{response.think or ''}{self.model.THINK_END}{response.answer}"})
            for observation in executor.tool_results()[n_before:]:
                messages.append(self.model.build_observation_message(observation))

        return build_result_record(record, normalize_calls(raw_calls), executor.tool_results(), final_response, raw_generation)

    @staticmethod
    def _turn_debug(response) -> Dict[str, Any]:
        """Everything the model actually decoded for one generate() turn -- the
        think block (reasoning before any <tool_call>), the raw text including
        the <tool_call>...</tool_call> block as generated, and the fully
        reassembled text (think + prefix + answer), for inspecting what the
        model produced before/around tool-call parsing."""
        return {
            "think": response.think,
            "answer_raw": response.answer,
            "full_raw_text": response.raw_text,
        }
