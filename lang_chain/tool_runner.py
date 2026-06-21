"""Execute LangChain tool calls from chat messages."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

MAX_TOOL_ROUNDS = 3
INTERNET_TOOL_NAME = "search_internet"
INTERNET_ALREADY_USED = (
    "Internet search was already performed in this turn. "
    "Answer using the previous search results."
)
INTERNET_NOTE = (
    "\n\n_Ответ дополнен данными из интернета (DuckDuckGo)._"
)


def run_with_tools(
    llm: ChatOpenAI,
    tools: list[BaseTool],
    messages: list[BaseMessage],
) -> tuple[str, bool]:
    """Call the LLM with tools until a final answer is produced."""
    tool_map = {tool.name: tool for tool in tools}
    bound_llm = llm.bind_tools(tools)
    current = list(messages)
    used_internet = False
    internet_search_done = False
    response: AIMessage | None = None

    for _ in range(MAX_TOOL_ROUNDS):
        response = bound_llm.invoke(current)
        if not response.tool_calls:
            break

        current.append(response)
        used_internet, internet_search_done = _run_tool_calls(
            response,
            tool_map,
            current,
            internet_search_done,
        )

    if response is None:
        return "", used_internet

    if response.tool_calls:
        return (
            "Could not finish tool calls. Please try again.",
            used_internet,
        )

    answer = str(response.content)
    if used_internet and INTERNET_NOTE not in answer:
        answer += INTERNET_NOTE
    return answer, used_internet


def _run_tool_calls(
    response: AIMessage,
    tool_map: dict[str, BaseTool],
    current: list[BaseMessage],
    internet_search_done: bool,
) -> tuple[bool, bool]:
    used_internet = False
    for call in response.tool_calls:
        name = call["name"]
        if name == INTERNET_TOOL_NAME:
            used_internet = True
            if internet_search_done:
                result = INTERNET_ALREADY_USED
            else:
                result = str(tool_map[name].invoke(call["args"]))
                internet_search_done = True
        else:
            result = str(tool_map[name].invoke(call["args"]))

        current.append(
            ToolMessage(
                content=result,
                tool_call_id=call["id"],
            )
        )
    return used_internet, internet_search_done
