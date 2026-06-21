"""RAG chat over Pinecone documents."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from lang_chain.config import Settings
from lang_chain.llm import build_chat_model
from lang_chain.store import LangChainPineconeStore
from lang_chain.tool_runner import run_with_tools
from lang_chain.tools import CHAT_TOOLS

MAX_HISTORY_TURNS = 5
EXIT_COMMANDS = frozenset({"exit", "quit", "/exit", "/quit"})

SYSTEM_PROMPT = """You are a helpful assistant with a knowledge base and tools.

Knowledge base context (prefer this when it is sufficient):
{context}

Tools:
- search_internet: search DuckDuckGo when the knowledge base lacks the answer
- get_currency_rate: get exchange rates between currency codes

Rules:
- Reply in the same language as the user's question.
- Prefer the knowledge base; use search_internet only when it is insufficient.
- If you use search_internet, clearly say that internet sources (DuckDuckGo) were used.
- Call search_internet at most once per user question.
- Use get_currency_rate for currency exchange questions."""


class ChatService:
    """Answer questions with retrieval-augmented generation and tools."""

    def __init__(
        self,
        store: LangChainPineconeStore,
        llm: ChatOpenAI,
        index_name: str,
        top_k: int = 5,
        namespace: str = "",
    ) -> None:
        self._store = store
        self._llm = llm
        self._index_name = index_name
        self._top_k = top_k
        self._namespace = namespace
        self._history: list[BaseMessage] = []

    def ask(self, question: str) -> str:
        """Retrieve context, call tools if needed, update dialog history."""
        context = self._build_context(question)
        messages = self._build_messages(context, question)
        answer, _ = run_with_tools(self._llm, CHAT_TOOLS, messages)
        self._append_history(question, answer)
        return answer

    def _build_context(self, question: str) -> str:
        results = self._store.search(
            index_name=self._index_name,
            query=question,
            top_k=self._top_k,
            namespace=self._namespace,
        )
        if not results:
            return "No relevant documents found."
        return "\n".join(
            f"- [score={item.score:.4f}] {item.text}" for item in results
        )

    def _build_messages(
        self,
        context: str,
        question: str,
    ) -> list[BaseMessage]:
        system = SystemMessage(
            content=SYSTEM_PROMPT.format(context=context),
        )
        return [system, *self._history, HumanMessage(content=question)]

    def _append_history(self, question: str, answer: str) -> None:
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))
        self._trim_history()

    def _trim_history(self) -> None:
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]


def build_chat_service(
    settings: Settings,
    index_name: str,
    top_k: int = 5,
    namespace: str = "",
) -> ChatService:
    """Factory for interactive chat."""
    return ChatService(
        store=LangChainPineconeStore(settings),
        llm=build_chat_model(settings),
        index_name=index_name,
        top_k=top_k,
        namespace=namespace,
    )


def is_exit_command(text: str) -> bool:
    """Return True when the user wants to leave chat mode."""
    return text.strip().lower() in EXIT_COMMANDS
