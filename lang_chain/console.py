"""Interactive console input with readline editing and history."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path

try:
    import readline
except ImportError:
    readline = None  # type: ignore[assignment]

DEFAULT_HISTORY_PATH = Path.home() / ".lang_chain_history"
MAX_HISTORY_SIZE = 1000


class ConsoleReader:
    """Read user input with line editing and persistent history."""

    def __init__(
        self,
        history_path: Path = DEFAULT_HISTORY_PATH,
    ) -> None:
        self._history_path = history_path
        self._readline_enabled = readline is not None and sys.stdin.isatty()
        if self._readline_enabled:
            self._setup_readline()

    def read_line(self, prompt: str = "You> ") -> str:
        """Read one line; arrows edit text and browse history."""
        try:
            return input(prompt)
        except EOFError:
            self.save_history()
            raise

    def drop_last_history_entry(self) -> None:
        """Remove the last history item (e.g. after empty submit)."""
        if not self._readline_enabled:
            return
        length = readline.get_current_history_length()
        if length > 0:
            readline.remove_history_item(length - 1)

    def save_history(self) -> None:
        """Persist input history to disk."""
        if not self._readline_enabled:
            return
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(self._history_path))

    def _setup_readline(self) -> None:
        readline.set_history_length(MAX_HISTORY_SIZE)
        self._load_history()
        atexit.register(self.save_history)

    def _load_history(self) -> None:
        try:
            readline.read_history_file(str(self._history_path))
        except FileNotFoundError:
            pass
