from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.react.agent import ReActSingleInputOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.tools.render import ToolsRenderer, render_text_description
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough

from config import settings
from app.llm import get_chat_model
from app.tools import SHOTGRID_TOOLS

logger = logging.getLogger("chat.agent")

SYSTEM_PROMPT = (
    "너는 ShotGrid 파이프라인 운영을 지원하는 어시스턴트다. "
    "사용자의 질문을 분석해 필요한 경우 제공된 ShotGrid 툴을 호출하고, "
    "데이터가 없으면 그 이유와 후속 조치를 안내한다. "
    "응답은 한국어로 작성하고, 핵심 상태/일정/담당자 정보를 강조하며 "
    "다음 행동 제안을 덧붙인다. "
    "첫 응답은 반드시 'Thought:'로 시작해라. "
    "도구를 사용할 때는 ReAct 형식을 엄격히 지켜라."
)


def create_chat_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: ReActSingleInputOutputParser | None = None,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: bool | list[str] = True,
) -> Runnable:
    """Create a ReAct agent that works with chat prompts (message placeholders)."""
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        msg = f"Prompt missing required variables: {missing_vars}"
        raise ValueError(msg)

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm
    output_parser = output_parser or ReActSingleInputOutputParser()
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | output_parser
    )


@lru_cache(maxsize=1)
def get_agent_executor() -> AgentExecutor:
    """Create (and cache) the LangChain agent executor for chat handling."""
    llm = get_chat_model()
    prompt_messages = [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
    ]

    provider = (settings.llm_provider or "").strip().lower()
    if provider == "openai":
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        agent = create_tool_calling_agent(llm=llm, tools=SHOTGRID_TOOLS, prompt=prompt)
        tool_mode = "tool-calling"
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PROMPT
                    + "\n\n사용 가능한 도구 목록:\n{tools}\n"
                    + "도구 호출 시 이름은 반드시 다음 중 하나여야 한다: {tool_names}.\n"
                    + "반드시 아래 형식을 그대로 사용하라 (예시 포함):\n"
                    + "Question: 질문 원문\n"
                    + "Thought: 사용할 전략 설명\n"
                    + "Action: 선택한 도구 이름 (예: get_projects_progress)\n"
                    + "Action Input: {{\"project_name\": \"프로젝트명\"}}\n"
                    + "Observation: ...\n"
                    + "Thought: 결과 요약\n"
                    + "Final Answer: 최종 응답\n"
                    + "형식은 대소문자와 콜론까지 정확히 지켜야 하며, Action을 쓰면 바로 다음 줄에 Action Input을 JSON 객체로 작성해야 한다.\n"
                    + "도구를 쓰지 않을 때는 Thought와 Final Answer 두 줄만 출력한다.",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Question: {input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
            ]
        )
        agent = create_chat_react_agent(llm=llm, tools=SHOTGRID_TOOLS, prompt=prompt)
        tool_mode = "react"

    logger.info(
        "Agent executor initialised with provider=%s model=%s tools=%d mode=%s",
        settings.llm_provider,
        settings.llm_model,
        len(SHOTGRID_TOOLS),
        tool_mode,
    )
    return AgentExecutor(
        agent=agent,
        tools=SHOTGRID_TOOLS,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=settings.agent_max_iterations,
    )
