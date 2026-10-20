"""LLM-judge prompts and a thin OpenAI wrapper shared by scorer.py."""

import json

ARGUMENT_VALUE_PROMPT = """You are judging a voice agent's tool-call arguments. The user SPOKE the request, so the agent heard it through speech recognition. The values below already failed an exact string match against the reference; decide whether the agent's values are still ACCEPTABLE.

ACCEPTABLE (score 1) -- the agent took the right value from what the user said, and only the surface form differs:
- Formatting: case, punctuation, spacing, underscores, filler words, numbers-as-strings, abbreviations ("St" = "Street"), equivalent phrasings ("50 to 100" = "50-100").
- Shorter or longer phrasing of the same concept (category "comedy" for "comedy shows", "Italian" for "Italian food").
- Misspelled or misheard versions of what the user said -- the kind of error speech recognition makes:
  - place, person, business or product names ("Burlington" for "Ballard", "Catherine" for "Katherine"). Judge names by SOUND, not by geography or identity: if the agent's name sounds similar (e.g. same first letter/syllable) and fills the same slot, it is a mishearing and acceptable, even if it happens to be another real place or person. Only a name that sounds unrelated (e.g. "Chicago" for "Seattle") is wrong;
  - spoken letter/number codes and IDs whose letters are misheard but whose digits are right ("p5" or "pto5" for "O-5", "lock5" for "LOC-5", "EBT145" for "EVT-145").
- For an ID-type argument (product_id, product_ref, ...) whose reference is the item NAME the user said, an ID the agent looked up for that same item (e.g. "ITEM-341" or "PROD-8841" from a search result) is acceptable, and vice versa.
- Extra arguments not in the reference, as long as they don't contradict it.

NOT ACCEPTABLE (score 0) -- the value is wrong or made up:
- A different number, date, time, quantity, price, or digits in a code/ID.
- A value with no basis in what the user said (fabricated / guessed / filled with a default or placeholder).
- A value that clearly means a different thing (a different cuisine, a different item, a different category), or the value the user retracted in a self-correction.
- A reference argument that is missing from the agent's call.

Return ONLY {{"score": 1}} or {{"score": 0}}.

User query: {query}
Reference call: {reference}
Agent's actual call: {actual}
"""

CLARIFY_PROMPT = """A voice agent decided it needed more information before calling a tool, instead of guessing.

Judge whether the agent's response asks a clarifying question covering the same missing information as the reference question (wording may differ).

Return ONLY {{"score": 1}} or {{"score": 0}}.

Reference clarifying question: {reference_question}
Agent's actual response: {actual_response}
"""

GROUNDING_PROMPT = """A voice agent called some tools and got back real results, then gave a spoken response.

Judge whether the response is a faithful report of those real results: no contradicted or invented values, and no result silently dropped that the user needed to hear.

Return ONLY {{"score": 1}} or {{"score": 0}}.

User query: {query}
Real tool calls and results: {tool_results}
Agent's final response: {final_response}
"""

ZERO_CALL_ACTION_PROMPT = """A voice agent decided not to call any tool for this request, and gave a spoken response instead.

Classify what the agent's response actually does:
- "no_call": the agent directly handles/answers the request without needing any tool -- nothing is missing and it isn't declining.
- "clarify": the agent asks the user for missing or ambiguous information it would need before it could act. This counts even if the response has no literal question mark (e.g. "Just let me know when you'd like to travel." is a clarify).
- "cannot_answer": the agent states it is unable to help with this request at all -- e.g. it lacks the tool/access/authority/capability, or the request is out of scope -- not because information is missing, but because it cannot do this kind of thing.

Return ONLY {{"action": "no_call"}}, {{"action": "clarify"}}, or {{"action": "cannot_answer"}}.

User query: {query}
Agent's response: {final_response}
"""

_client = None

# How many judge calls failed (API error / unparseable or invalid answer) --
# distinct from "no LLM configured"; evaluate.py reports it per record.
judge_failures = 0


def _client_or_none():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI()
        except Exception:
            _client = False
    return _client or None


def llm_judge_score(prompt: str) -> "int | None":
    """Returns 1/0, or None if no LLM is configured / the call failed."""
    client = _client_or_none()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        return int(bool(json.loads(text)["score"]))
    except Exception:
        global judge_failures
        judge_failures += 1
        return None


def classify_zero_call_action(query: str, final_response: str) -> "str | None":
    """Classifies a no-tool-call response as no_call/clarify/cannot_answer.
    Returns None if no LLM is configured / the call failed / the answer
    wasn't one of the three labels."""
    client = _client_or_none()
    if client is None:
        return None
    prompt = ZERO_CALL_ACTION_PROMPT.format(query=query, final_response=final_response)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        action = json.loads(text)["action"]
    except Exception:
        action = None
    if action in ("no_call", "clarify", "cannot_answer"):
        return action
    global judge_failures
    judge_failures += 1
    return None
