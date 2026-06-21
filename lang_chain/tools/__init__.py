"""LangChain tools for chat."""

from lang_chain.tools.currency import get_currency_rate
from lang_chain.tools.web_search import search_internet

CHAT_TOOLS = [search_internet, get_currency_rate]
