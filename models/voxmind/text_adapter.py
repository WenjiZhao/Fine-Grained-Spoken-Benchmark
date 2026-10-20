"""VoxMind text adapter: dataset record -> normalized result dict, feeding
the ground-truth transcript directly as text instead of the TTS audio.

Same two-turn ReAct loop as adapter.py's VoxMindAdapter, just with the user
message built from record["transcript"] (content type "text") instead of
resolved_audio_path(record) (content type "audio"). Useful for isolating
ASR/audio-understanding errors from tool-calling errors: if VoxMind gets the
right answer here but not from audio, the gap is in audio understanding, not
tool-calling.
"""

import sys
from pathlib import Path
from typing import Any, Dict

BENCH_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from runtime import DEFAULT_SYSTEM_PROMPT, VoxMind  # noqa: E402

from models.common.dataset import available_tool_names, tool_schemas_for  # noqa: E402
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


class VoxMindTextAdapter:
    def __init__(self, checkpoint: str, device: str = "cuda", extra_context: dict | None = None):
        self.model = VoxMind(checkpoint, device=device)
        self.extra_context = extra_context or {}

    def run(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        tools = tool_schemas_for(record, all_schemas)
        system_prompt = self.model.build_system_prompt(
            DEFAULT_SYSTEM_PROMPT, tools, extra_context=self.extra_context
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": record["transcript"]}]},
        ]

        first = self.model.generate(
            messages, response_mode="text", post_think_prefix=POST_THINK_PREFIX, **GENERATION_KWARGS
        )
        raw_calls = self.model.parse_tool_calls(first.answer)
        raw_generation = {"turn1": self._turn_debug(first)}

        executor = MockToolExecutor(allowed_tools=available_tool_names(record))  # fresh per record, per executor.py's own contract
        for call in raw_calls:
            executor.call(call.get("name", ""), **(call.get("arguments") or {}))
        tool_results = executor.tool_results()

        if raw_calls:
            messages.append({"role": "assistant", "content": f"{self.model.THINK_START}\n{first.think or ''}{self.model.THINK_END}{first.answer}"})
            for observation in tool_results:
                messages.append(self.model.build_observation_message(observation))
            second = self.model.generate(
                messages, response_mode="text", post_think_prefix=POST_THINK_PREFIX, **GENERATION_KWARGS
            )
            final_response = second.answer
            raw_generation["turn2"] = self._turn_debug(second)
        else:
            final_response = first.answer

        return build_result_record(record, normalize_calls(raw_calls), tool_results, final_response, raw_generation)

    @staticmethod
    def _turn_debug(response) -> Dict[str, Any]:
        return {
            "think": response.think,
            "answer_raw": response.answer,
            "full_raw_text": response.raw_text,
        }
