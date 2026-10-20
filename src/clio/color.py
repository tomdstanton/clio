import argparse
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar, Self


def _detect_color_support() -> bool:
    r"""Detect whether the current terminal environment supports ANSI color output.

    Respects `NO_COLOR`, `FORCE_COLOR`, `CLICOLOR_FORCE`, `TERM=dumb`, and TTY status.

    Returns:
        bool: `True` if ANSI styling should be enabled by default.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


@dataclass(frozen=True, slots=True)
class Style:
    r"""Immutable, composable ANSI style specification with pre-compiled escape prefix.

    Pre-computes the combined ANSI escape sequence at construction time so repeated
    application via `style(text)` requires zero list building or string joining.

    Attributes:
        fg (str): Foreground ANSI escape sequence.
        bg (str): Background ANSI escape sequence.
        bold (bool): Apply bold modifier.
        dim (bool): Apply dim/faint modifier.
        italic (bool): Apply italic modifier.
        underline (bool): Apply underline modifier.
        blink (bool): Apply blink modifier.
        reverse (bool): Apply reverse video modifier.
        hidden (bool): Apply hidden text modifier.
        strikethrough (bool): Apply strikethrough modifier.
        prefix (str): Pre-compiled combined ANSI prefix sequence.
    """

    fg: str = ""
    bg: str = ""
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    blink: bool = False
    reverse: bool = False
    hidden: bool = False
    strikethrough: bool = False
    prefix: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        r"""Pre-compile the combined ANSI escape sequence."""
        seq: list[str] = []
        if self.bold:
            seq.append("\033[1m")
        if self.dim:
            seq.append("\033[2m")
        if self.italic:
            seq.append("\033[3m")
        if self.underline:
            seq.append("\033[4m")
        if self.blink:
            seq.append("\033[5m")
        if self.reverse:
            seq.append("\033[7m")
        if self.hidden:
            seq.append("\033[8m")
        if self.strikethrough:
            seq.append("\033[9m")
        if self.fg:
            seq.append(self.fg)
        if self.bg:
            seq.append(self.bg)
        object.__setattr__(self, "prefix", "".join(seq))

    def __call__(self, text: str) -> str:
        r"""Format text using the pre-compiled ANSI prefix.

        Args:
            text (str): Input text to style.

        Returns:
            str: Styled text if `Colors.ENABLED` is active, otherwise original text.
        """
        if not Colors.ENABLED or not self.prefix or not text:
            return text
        return f"{self.prefix}{text}\033[0m"

    def __or__(self, other: Style) -> Self:
        r"""Merge two `Style` specifications into a new immutable `Style`.

        Args:
            other (Style): Style specification whose attributes override or augment `self`.

        Returns:
            Self: Newly composed `Style` instance.
        """
        return type(self)(
            fg=other.fg or self.fg,
            bg=other.bg or self.bg,
            bold=self.bold or other.bold,
            dim=self.dim or other.dim,
            italic=self.italic or other.italic,
            underline=self.underline or other.underline,
            blink=self.blink or other.blink,
            reverse=self.reverse or other.reverse,
            hidden=self.hidden or other.hidden,
            strikethrough=self.strikethrough or other.strikethrough,
        )


class Colors:
    r"""Namespace for ANSI terminal color escape sequences and styling helpers.

    Provides constants, 256-color/TrueColor builders, and ergonomic helper methods
    for terminal text formatting. This class is designed as a pure namespace and
    cannot be instantiated.

    Attributes:
        ENABLED (bool): Flag indicating whether ANSI color output is active.
        RESET (str): ANSI reset escape sequence.
        BOLD (str): ANSI bold text formatting sequence.
        DIM (str): ANSI dim/faint text formatting sequence.
        ITALIC (str): ANSI italic text formatting sequence.
        UNDERLINE (str): ANSI underline text formatting sequence.
        BLINK (str): ANSI blink text formatting sequence.
        REVERSE (str): ANSI reverse/invert video formatting sequence.
        HIDDEN (str): ANSI hidden/concealed text formatting sequence.
        STRIKETHROUGH (str): ANSI strikethrough text formatting sequence.
    """

    ENABLED: ClassVar[bool] = _detect_color_support()

    _ANSI_RE: ClassVar[re.Pattern[str]] = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self) -> None:
        r"""Prevent instantiation of namespace class.

        Raises:
            TypeError: Always raised when instantiated.
        """
        raise TypeError("The Colors class is a namespace and cannot be instantiated.")

    # --- Text Formatting Modifiers ---
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    INVERT = "\033[7m"
    HIDDEN = "\033[8m"
    STRIKETHROUGH = "\033[9m"

    # --- Standard Foreground Colors (30-37) ---
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # --- Bright Foreground Colors (90-97) ---
    BRIGHT_BLACK = "\033[90m"
    GRAY = "\033[90m"
    GREY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # --- Pre-combined Bold Foreground Colors ---
    BOLD_BLACK = "\033[1;30m"
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_MAGENTA = "\033[1;35m"
    BOLD_CYAN = "\033[1;36m"
    BOLD_WHITE = "\033[1;37m"

    # --- Standard Background Colors (40-47) ---
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # --- Bright Background Colors (100-107) ---
    BG_BRIGHT_BLACK = "\033[100m"
    BG_GRAY = "\033[100m"
    BG_GREY = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"

    @classmethod
    def wrap(cls, text: str, *styles: str) -> str:
        r"""Wrap text in specified ANSI styling escape sequences.

        Uses a single-style fast path to avoid tuple joining overhead and appends
        the reset sequence at the end when color output is enabled.

        Args:
            text (str): Input text string to format.
            *styles (str): ANSI style escape sequences to apply.

        Returns:
            str: Styled text string, or original text if ANSI color is disabled or no styles are given.
        """
        if not cls.ENABLED or not styles or not text:
            return text
        prefix = styles[0] if len(styles) == 1 else "".join(styles)
        if not prefix:
            return text
        return f"{prefix}{text}{cls.RESET}"

    @classmethod
    def style(
        cls,
        text: str,
        *,
        fg: str | None = None,
        bg: str | None = None,
        bold: bool = False,
        dim: bool = False,
        italic: bool = False,
        underline: bool = False,
        blink: bool = False,
        reverse: bool = False,
        hidden: bool = False,
        strikethrough: bool = False,
    ) -> str:
        r"""Apply multiple declarative style flags and foreground/background colors to text.

        Args:
            text (str): Input text string to format.
            fg (str | None): Foreground ANSI color sequence.
            bg (str | None): Background ANSI color sequence.
            bold (bool): Apply bold modifier.
            dim (bool): Apply dim/faint modifier.
            italic (bool): Apply italic modifier.
            underline (bool): Apply underline modifier.
            blink (bool): Apply blink modifier.
            reverse (bool): Apply reverse video modifier.
            hidden (bool): Apply hidden text modifier.
            strikethrough (bool): Apply strikethrough modifier.

        Returns:
            str: Styled text string, or original text if ANSI color is disabled.
        """
        if not cls.ENABLED or not text:
            return text
        seq: list[str] = []
        if bold:
            seq.append(cls.BOLD)
        if dim:
            seq.append(cls.DIM)
        if italic:
            seq.append(cls.ITALIC)
        if underline:
            seq.append(cls.UNDERLINE)
        if blink:
            seq.append(cls.BLINK)
        if reverse:
            seq.append(cls.REVERSE)
        if hidden:
            seq.append(cls.HIDDEN)
        if strikethrough:
            seq.append(cls.STRIKETHROUGH)
        if fg:
            seq.append(fg)
        if bg:
            seq.append(bg)
        if not seq:
            return text
        return f"{''.join(seq)}{text}{cls.RESET}"

    # --- 256-Color & 24-bit TrueColor Builders ---
    @staticmethod
    def fg256(code: int) -> str:
        r"""Return 8-bit (256-color) foreground ANSI escape sequence.

        Args:
            code (int): Color palette index in `0..255`.

        Returns:
            str: ANSI escape sequence for the 256-color foreground.

        Raises:
            ValueError: If `code` is outside `0..255`.
        """
        if not 0 <= code <= 255:
            raise ValueError(f"256-color code must be between 0 and 255, got {code}")
        return f"\033[38;5;{code}m"

    @staticmethod
    def bg256(code: int) -> str:
        r"""Return 8-bit (256-color) background ANSI escape sequence.

        Args:
            code (int): Color palette index in `0..255`.

        Returns:
            str: ANSI escape sequence for the 256-color background.

        Raises:
            ValueError: If `code` is outside `0..255`.
        """
        if not 0 <= code <= 255:
            raise ValueError(f"256-color code must be between 0 and 255, got {code}")
        return f"\033[48;5;{code}m"

    @staticmethod
    def rgb(r: int, g: int, b: int, *, background: bool = False) -> str:
        r"""Return 24-bit TrueColor RGB ANSI escape sequence.

        Args:
            r (int): Red channel intensity (`0..255`).
            g (int): Green channel intensity (`0..255`).
            b (int): Blue channel intensity (`0..255`).
            background (bool): If `True`, generate background sequence instead of foreground.

        Returns:
            str: ANSI TrueColor escape sequence.

        Raises:
            ValueError: If any RGB channel is outside `0..255`.
        """
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError(f"RGB channels must each be between 0 and 255, got ({r}, {g}, {b})")
        layer = 48 if background else 38
        return f"\033[{layer};2;{r};{g};{b}m"

    @classmethod
    def hex(cls, hex_color: str, *, background: bool = False) -> str:
        r"""Return 24-bit TrueColor ANSI escape sequence from a hex color string.

        Supports both 6-digit (`"#ff5500"`, `"ff5500"`) and 3-digit (`"#f50"`, `"f50"`) hex strings.

        Args:
            hex_color (str): Hexadecimal RGB string.
            background (bool): If `True`, generate background sequence instead of foreground.

        Returns:
            str: ANSI TrueColor escape sequence.

        Raises:
            ValueError: If `hex_color` is not a valid 3- or 6-character hex string.
        """
        cleaned = hex_color.strip().lstrip("#")
        if len(cleaned) == 3:
            cleaned = "".join(ch * 2 for ch in cleaned)
        if len(cleaned) != 6:
            raise ValueError(f"Invalid hex color string: '{hex_color}'")
        try:
            r = int(cleaned[0:2], 16)
            g = int(cleaned[2:4], 16)
            b = int(cleaned[4:6], 16)
        except ValueError as exc:
            raise ValueError(f"Invalid hex color string: '{hex_color}'") from exc
        return cls.rgb(r, g, b, background=background)

    # --- ANSI Stripping & Measurement Utilities ---
    @classmethod
    def strip_ansi(cls, text: str) -> str:
        r"""Remove all ANSI styling escape sequences from text.

        Args:
            text (str): String potentially containing ANSI escape sequences.

        Returns:
            str: Plain text with all ANSI sequences stripped.
        """
        if "\033[" not in text:
            return text
        return cls._ANSI_RE.sub("", text)

    @classmethod
    def visible_len(cls, text: str) -> int:
        r"""Compute the visible character length of text excluding ANSI escape codes.

        Args:
            text (str): String potentially containing ANSI escape sequences.

        Returns:
            int: Visible character count.
        """
        return len(cls.strip_ansi(text))

    # --- Ergonomic Style & Color Helpers ---
    @classmethod
    def bold(cls, text: str) -> str:
        r"""Format text in bold."""
        return cls.wrap(text, cls.BOLD)

    @classmethod
    def dim(cls, text: str) -> str:
        r"""Format text in dim/faint style."""
        return cls.wrap(text, cls.DIM)

    @classmethod
    def italic(cls, text: str) -> str:
        r"""Format text in italics."""
        return cls.wrap(text, cls.ITALIC)

    @classmethod
    def underline(cls, text: str) -> str:
        r"""Format text with underline."""
        return cls.wrap(text, cls.UNDERLINE)

    @classmethod
    def strikethrough(cls, text: str) -> str:
        r"""Format text with strikethrough."""
        return cls.wrap(text, cls.STRIKETHROUGH)

    @classmethod
    def black(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in black (or bold black)."""
        return cls.wrap(text, cls.BOLD_BLACK if bold else cls.BLACK)

    @classmethod
    def red(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in red (or bold red)."""
        return cls.wrap(text, cls.BOLD_RED if bold else cls.RED)

    @classmethod
    def green(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in green (or bold green)."""
        return cls.wrap(text, cls.BOLD_GREEN if bold else cls.GREEN)

    @classmethod
    def yellow(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in yellow (or bold yellow)."""
        return cls.wrap(text, cls.BOLD_YELLOW if bold else cls.YELLOW)

    @classmethod
    def blue(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in blue (or bold blue)."""
        return cls.wrap(text, cls.BOLD_BLUE if bold else cls.BLUE)

    @classmethod
    def magenta(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in magenta (or bold magenta)."""
        return cls.wrap(text, cls.BOLD_MAGENTA if bold else cls.MAGENTA)

    @classmethod
    def cyan(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in cyan (or bold cyan)."""
        return cls.wrap(text, cls.BOLD_CYAN if bold else cls.CYAN)

    @classmethod
    def white(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in white (or bold white)."""
        return cls.wrap(text, cls.BOLD_WHITE if bold else cls.WHITE)

    @classmethod
    def gray(cls, text: str, *, bold: bool = False) -> str:
        r"""Format text in gray/bright black."""
        return cls.wrap(text, cls.BOLD, cls.GRAY) if bold else cls.wrap(text, cls.GRAY)

    # --- Semantic CLI Presets ---
    @classmethod
    def success(cls, text: str) -> str:
        r"""Format text with semantic success styling (bold green)."""
        return cls.wrap(text, cls.BOLD_GREEN)

    @classmethod
    def warning(cls, text: str) -> str:
        r"""Format text with semantic warning styling (bold yellow)."""
        return cls.wrap(text, cls.BOLD_YELLOW)

    @classmethod
    def error(cls, text: str) -> str:
        r"""Format text with semantic error styling (bold red)."""
        return cls.wrap(text, cls.BOLD_RED)

    @classmethod
    def info(cls, text: str) -> str:
        r"""Format text with semantic informational styling (bold cyan)."""
        return cls.wrap(text, cls.BOLD_CYAN)

    @classmethod
    def muted(cls, text: str) -> str:
        r"""Format text with semantic muted styling (dim)."""
        return cls.wrap(text, cls.DIM)


class ColorHelpFormatter(argparse.RawTextHelpFormatter):
    r"""Custom argparse help formatter adding ANSI colors and layout enhancements.

    Inherits from `argparse.RawTextHelpFormatter` to preserve
    raw multiline descriptions while adding styled section headings and usage placeholders.
    """

    _SUBCOMMAND_SET_RE: ClassVar[re.Pattern[str]] = re.compile(r"\{[a-zA-Z0-9_,\.-]+\}")

    def _format_usage(
        self,
        usage: str | None,
        actions: Iterable[argparse.Action],
        groups: Iterable[argparse._ArgumentGroup],
        prefix: str | None,
    ) -> str:
        r"""Format usage syntax string with ANSI color highlights.

        Args:
            usage (str | None): Custom usage message pattern.
            actions (Iterable[argparse.Action]): Sequence of argument actions.
            groups (Iterable[argparse._ArgumentGroup]): Sequence of argument groups.
            prefix (str | None): Usage line prefix string (defaults to "usage: ").

        Returns:
            str: Colorized usage string.
        """
        positionals: list[argparse.Action] = []
        has_optionals = False
        for action in actions:
            if action.option_strings:
                has_optionals = True
            else:
                positionals.append(action)

        result = super()._format_usage(usage, positionals, groups, prefix)  # type: ignore

        # Replace the subcommand set {add,install,...} with [subcommand]
        result = self._SUBCOMMAND_SET_RE.sub(Colors.info("[subcommand]"), result)

        actual_prefix = prefix if prefix is not None else "usage: "
        target = f"{actual_prefix}{self._prog}"

        if result.startswith(target):
            # Inject [options] cleanly if any optional actions exist
            if has_optionals:
                colored_options = Colors.info("[options]")
                result = result.replace(target, f"{target} {colored_options}", 1)
            # Colorize prefix
            result = result.replace(actual_prefix, Colors.info(actual_prefix), 1)

        return result

    def start_section(self, heading: str | None) -> None:
        r"""Start a new help output section with a colorized heading.

        Args:
            heading (str | None): Title heading string for the section.
        """
        if heading:
            heading = Colors.info(heading)
        super().start_section(heading)

    def _format_action(self, action: argparse.Action) -> str:
        r"""Format help output text for a single argument action.

        Args:
            action (argparse.Action): Argument action instance being formatted.

        Returns:
            str: Formatted action description string.
        """
        result = super()._format_action(action)
        # Remove the set string (e.g. {add,install,...}) from the subcommands header
        if isinstance(action, argparse._SubParsersAction):
            lines = result.split("\n", 1)
            if len(lines) > 1:
                result = lines[1]
        return result
