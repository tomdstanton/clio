import functools
import inspect
import os
import sys
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self, TextIO, cast

from clio.agent import _ACTIVE_MESSAGE_SINK, AgentMessage, _detect_agent_mode
from clio.color import Colors, Style

_DEFAULT_SPINNER_STYLE = Style(fg=Colors.CYAN, bold=True)


@dataclass(frozen=True, slots=True)
class SpinnerConfig:
    r"""Immutable visual and timing specification for a terminal [`Spinner`][clio.spinner.Spinner].

    Attributes:
        frames (tuple[str, ...]): Ordered sequence of animation glyph frames.
        interval (float): Frame refresh interval in seconds.
        success_symbol (str): Glyph rendered upon successful completion.
        error_symbol (str): Glyph rendered when an exception or explicit failure occurs.
        style (Style): ANSI style applied to active animation frames.
    """

    frames: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    interval: float = 0.08
    success_symbol: str = "✔"
    error_symbol: str = "✖"
    style: Style = _DEFAULT_SPINNER_STYLE


_DEFAULT_SPINNER_CONFIG = SpinnerConfig()


class Spinner:
    r"""Zero-dependency sync and async terminal spinner with automatic agent/TTY adaptation.

    Supports both synchronous (`with Spinner(...)`) and asynchronous (`async with Spinner(...)`)
    context management using a lightweight `threading.Event` daemon timer so animations remain
    responsive even during GIL-releasing native C/Rust/Polars/Numba calls.

    Automatically disables carriage-return (`\r`) and cursor-hide animations when `CLIO_AGENT=1`
    is set, when the output stream is not an interactive TTY, or when `--json` / MCP capture
    is active.
    """

    __slots__ = (
        "_agent_mode",
        "_config",
        "_explicit_outcome",
        "_interactive",
        "_lock",
        "_message",
        "_show_elapsed",
        "_start_ns",
        "_stop_event",
        "_stream",
        "_thread",
    )

    def __init__(
        self,
        message: str,
        *,
        config: SpinnerConfig = _DEFAULT_SPINNER_CONFIG,
        stream: TextIO | None = None,
        agent_mode: bool | None = None,
        show_elapsed: bool = True,
    ) -> None:
        r"""Initialize a terminal spinner.

        Args:
            message (str): Initial status message displayed alongside the spinner.
            config (SpinnerConfig): Visual frames, timing, and styling specification.
            stream (TextIO | None): Target output stream (defaults to `sys.stderr`).
            agent_mode (bool | None): Override `CLIO_AGENT=1` detection when provided.
            show_elapsed (bool): Whether to append elapsed seconds on completion.
        """
        self._message = message
        self._config = config
        self._stream = stream
        self._agent_mode = agent_mode
        self._show_elapsed = show_elapsed
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ns: int = 0
        self._interactive: bool = False
        self._explicit_outcome: tuple[bool, str] | None = None

    def _resolve_stream(self) -> TextIO:
        return sys.stderr if self._stream is None else self._stream

    def _is_interactive_tty(self, stream: TextIO) -> bool:
        if _ACTIVE_MESSAGE_SINK.get() is not None:
            return False
        is_agent = _detect_agent_mode() if self._agent_mode is None else self._agent_mode
        if is_agent:
            return False
        if os.environ.get("TERM") == "dumb":
            return False
        return hasattr(stream, "isatty") and stream.isatty()

    def update(self, message: str) -> None:
        r"""Update the active spinner message text in a thread-safe manner.

        Args:
            message (str): Replacement status message text.
        """
        with self._lock:
            self._message = message

    def succeed(self, message: str | None = None) -> None:
        r"""Mark the spinner outcome as successful, optionally overriding its final message.

        Args:
            message (str | None): Optional final completion text.
        """
        with self._lock:
            final_msg = self._message if message is None else message
            self._explicit_outcome = (True, final_msg)

    def fail(self, message: str | None = None) -> None:
        r"""Mark the spinner outcome as failed, optionally overriding its final message.

        Args:
            message (str | None): Optional final failure text.
        """
        with self._lock:
            final_msg = self._message if message is None else message
            self._explicit_outcome = (False, final_msg)

    def _render_loop(self, stream: TextIO) -> None:
        frames = self._config.frames or ("-",)
        frame_count = len(frames)
        interval = self._config.interval
        style = self._config.style
        idx = 0

        while not self._stop_event.is_set():
            glyph = style(frames[idx % frame_count])
            with self._lock:
                msg = self._message
                try:
                    stream.write(f"\r\033[2K{glyph} {msg}")
                    stream.flush()
                except OSError:
                    break
            idx += 1
            if self._stop_event.wait(interval):
                break

    def _format_elapsed(self) -> str:
        if not self._show_elapsed or self._start_ns == 0:
            return ""
        elapsed_s = max(0.0, (time.perf_counter_ns() - self._start_ns) / 1_000_000_000.0)
        return f" ({elapsed_s:.2f}s)"

    def __enter__(self) -> Self:
        r"""Start the spinner context."""
        self._start_ns = time.perf_counter_ns()
        self._stop_event.clear()
        self._explicit_outcome = None
        stream = self._resolve_stream()
        self._interactive = self._is_interactive_tty(stream)

        if self._interactive:
            with self._lock:
                stream.write("\033[?25l")
                stream.flush()
            self._thread = threading.Thread(
                target=self._render_loop,
                args=(stream,),
                name="clio-spinner",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        r"""Stop the spinner, restore terminal state, and emit final status."""
        stream = self._resolve_stream()
        try:
            if self._interactive:
                self._stop_event.set()
                if self._thread is not None:
                    self._thread.join(timeout=1.0)
                    self._thread = None
        finally:
            if self._interactive:
                with self._lock:
                    try:
                        stream.write("\r\033[2K\033[?25h")
                        stream.flush()
                    except OSError:
                        pass

        with self._lock:
            if self._explicit_outcome is not None:
                ok, final_msg = self._explicit_outcome
            else:
                ok = exc_type is None
                final_msg = self._message

        elapsed_suffix = self._format_elapsed()
        sink = _ACTIVE_MESSAGE_SINK.get()
        if sink is not None:
            clean_msg = Colors.strip_ansi(final_msg).strip()
            level = "info" if ok else "error"
            sink.append(AgentMessage(level=level, text=f"{clean_msg}{elapsed_suffix}"))
            return

        symbol = Colors.success(self._config.success_symbol) if ok else Colors.error(self._config.error_symbol)
        styled_elapsed = Colors.muted(elapsed_suffix) if elapsed_suffix else ""
        try:
            stream.write(f"{symbol} {final_msg}{styled_elapsed}\n")
            stream.flush()
        except OSError:
            pass

    async def __aenter__(self) -> Self:
        r"""Enter the spinner from an asynchronous `async with` block."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        r"""Exit the spinner from an asynchronous `async with` block."""
        self.__exit__(exc_type, exc_val, exc_tb)


def spinner[**P, R](
    message: str,
    *,
    config: SpinnerConfig = _DEFAULT_SPINNER_CONFIG,
    show_elapsed: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    r"""Decorate a synchronous or asynchronous callable with a live terminal [`Spinner`][clio.spinner.Spinner].

    Detects `inspect.iscoroutinefunction` once at decoration time and preserves `__wrapped__`
    so `annotationlib` and `DocstringParser` continue to inspect the underlying signature cleanly.

    Args:
        message (str): Status message displayed while the callable runs.
        config (SpinnerConfig): Optional visual and timing configuration.
        show_elapsed (bool): Whether to display elapsed execution time on completion.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: Type-preserving function or coroutine decorator.
    """

    def _decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                async with Spinner(message, config=config, show_elapsed=show_elapsed):
                    return await fn(*args, **kwargs)

            return cast(Callable[P, R], _async_wrapper)

        @functools.wraps(fn)
        def _sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with Spinner(message, config=config, show_elapsed=show_elapsed):
                return fn(*args, **kwargs)

        return _sync_wrapper

    return _decorator
