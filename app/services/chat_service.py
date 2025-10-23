from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain.schema import AgentAction

from app.agents import get_agent_executor

logger = logging.getLogger("chat.service")


def _parse_possible_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _serialise_tool_step(step: Tuple[AgentAction, Any]) -> Dict[str, Any]:
    action, observation = step
    parsed_observation = _parse_possible_json(observation)
    return {
        "output": parsed_observation,
    }


def run_chat(message: str, *, thread_id: str) -> Dict[str, Any]:
    """Execute the chat agent and return structured data for the HTTP layer."""
    executor = get_agent_executor()
    try:
        result = executor.invoke(
            {"input": message},
            config={"tags": [thread_id]},
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Chat agent execution failed thread=%s", thread_id)
        return {
            "reply": "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            "tool_calls": [],
            "error": str(exc),
        }

    output_text = result.get("output", "")
    steps = result.get("intermediate_steps") or []
    tool_calls: List[Dict[str, Any]] = [_serialise_tool_step(step) for step in steps]

    tool_calls = [entry["output"] for entry in tool_calls if "output" in entry]

    return {
        "reply": output_text,
        "tool_calls": tool_calls,
        "error": None,
    }
