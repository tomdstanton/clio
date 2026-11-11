import enum
import os
import shutil
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

from clio.color import BoxStyle, Colors, Style
from clio.docstr import ParameterSpec, _parse_bool

type EagerAction = Literal["help", "version", "schema", "help_md", "skill", "mcp"]


class CliError(Exception):
    r"""Base exception for expected CLI application errors.

    Exceptions inheriting from `CliError` (or passed via `catch_exceptions` to [`Cli`][clio.cli.Cli])
    are caught gracefully by the runner, printed to `sys.stderr` with formatting, and exit with
    a non-zero status code without dumping a raw Python traceback.
    """

    __slots__ = ("exit_code",)

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ParseError(CliError):
    r"""Structured CLI token, option, or subcommand parsing error (exit status 2).

    Attributes:
        usage (str): Compact CLI usage synopsis for the failing command level.
        suggestions (tuple[str, ...]): Fuzzy-matched flag, subcommand, or choice suggestions.
        command_path (str): Resolved program/subcommand label where parsing failed.
    """

    __slots__ = ("command_path", "suggestions", "usage")

    def __init__(
        self,
        message: str,
        *,
        usage: str = "",
        suggestions: tuple[str, ...] = (),
        command_path: str = "",
    ) -> None:
        super().__init__(message, exit_code=2)
        self.usage = usage
        self.suggestions = suggestions
        self.command_path = command_path


def _is_negative_number(token: str) -> bool:
    r"""Return `True` if `token` is a negative integer or floating-point literal (e.g. `-42`, `-1.5`, `-.5`)."""
    if len(token) < 2 or token[0] != "-" or token[1] not in "0123456789.":
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def _fuzzy_suggest(candidate: str, pool: Sequence[str]) -> tuple[str, ...]:
    r"""Lazily compute close string matches via `difflib.get_close_matches` on error paths only."""
    if not candidate or not pool:
        return ()
    from difflib import get_close_matches

    return tuple(get_close_matches(candidate, pool))


def _clone_default(val: Any) -> Any:
    r"""Return a shallow copy of mutable collection defaults so runs cannot mutate cached specs."""
    if isinstance(val, list):
        return list(val)
    if isinstance(val, set):
        return set(val)
    if isinstance(val, dict):
        return dict(val)
    return val


@dataclass(frozen=True, slots=True)
class OptionBinding:
    r"""Immutable flag-to-parameter entry inside a compiled [`ScanPlan`][clio.parser.ScanPlan].

    Attributes:
        spec (ParameterSpec): Target parameter specification bound to this flag.
        implicit_bool (bool | None): Implicit boolean value (`True` for `--flag`, `False` for `--no-flag`),
            or `None` when the option consumes explicit argument token(s).
    """

    spec: ParameterSpec
    implicit_bool: bool | None = None


@dataclass(frozen=True, slots=True)
class ScanResult:
    r"""Immutable result of scanning a command's CLI token slice via [`ScanPlan.parse`][clio.parser.ScanPlan.parse].

    Attributes:
        values (Mapping[str, Any]): Coerced parameter values keyed by Python parameter name.
        verbose (bool): Whether `-V` / `--verbose` was requested.
        json_mode (bool): Whether built-in `--json` output mode was requested.
        dry_run (bool): Whether built-in `--dry-run` validation mode was requested.
        eager_action (EagerAction | None): Eager introspection trigger (`"help"`, `"version"`,
            `"schema"`, `"help_md"`, `"skill"`, or `"mcp"`), if encountered before `--`.
    """

    values: Mapping[str, Any]
    verbose: bool = False
    json_mode: bool = False
    dry_run: bool = False
    eager_action: EagerAction | None = None


@dataclass(frozen=True, slots=True)
class ScanPlan:
    r"""Pre-compiled, immutable $O(1)$ CLI token scanner for a command's `ParameterSpec` sequence.

    Attributes:
        positionals (tuple[ParameterSpec, ...]): Ordered positional parameter specifications.
        options (tuple[ParameterSpec, ...]): Ordered `--flag` option parameter specifications.
        flag_map (Mapping[str, OptionBinding]): Read-only lookup mapping every short, long, and `--no-*`
            flag string to its `OptionBinding`.
        known_flags (tuple[str, ...]): All registered user option flags for fuzzy error suggestions.
    """

    positionals: tuple[ParameterSpec, ...]
    options: tuple[ParameterSpec, ...]
    flag_map: Mapping[str, OptionBinding]
    known_flags: tuple[str, ...]

    @classmethod
    def compile(cls, specs: Sequence[ParameterSpec]) -> Self:
        r"""Compile a stateless `ScanPlan` from an ordered sequence of `ParameterSpec` items.

        Args:
            specs (Sequence[ParameterSpec]): Parameter specifications to index.

        Returns:
            Self: Compiled `ScanPlan` instance.
        """
        positionals: list[ParameterSpec] = []
        options: list[ParameterSpec] = []
        flag_dict: dict[str, OptionBinding] = {}

        for spec in specs:
            if spec.is_positional:
                positionals.append(spec)
                continue

            options.append(spec)
            if spec.annotation is bool:
                pos_binding = OptionBinding(spec=spec, implicit_bool=True)
                for flag in spec.option_flags:
                    flag_dict[flag] = pos_binding
                neg = spec.negated_flag
                if neg is not None and neg not in flag_dict:
                    flag_dict[neg] = OptionBinding(spec=spec, implicit_bool=False)
            else:
                val_binding = OptionBinding(spec=spec, implicit_bool=None)
                for flag in spec.option_flags:
                    flag_dict[flag] = val_binding

        return cls(
            positionals=tuple(positionals),
            options=tuple(options),
            flag_map=types.MappingProxyType(flag_dict),
            known_flags=tuple(flag_dict.keys()),
        )

    def _builtin_flag_pool(
        self,
        *,
        allow_version: bool,
        enable_agent_flags: bool,
        allow_mcp: bool,
    ) -> tuple[str, ...]:
        pool: list[str] = list(self.known_flags)
        for g_flag in ("-h", "--help", "-V", "--verbose"):
            if g_flag not in self.flag_map:
                pool.append(g_flag)
        if allow_version:
            for v_flag in ("-v", "--version"):
                if v_flag not in self.flag_map:
                    pool.append(v_flag)
        if enable_agent_flags:
            for a_flag in ("--json", "--dry-run", "--schema", "--help-md", "--skill"):
                if a_flag not in self.flag_map:
                    pool.append(a_flag)
            if allow_mcp and "--mcp" not in self.flag_map:
                pool.append("--mcp")
        return tuple(pool)

    def _coerce_with_parse_error(
        self,
        spec: ParameterSpec,
        raw_val: Any,
        *,
        label: str,
        prog: str,
        usage: str,
    ) -> Any:
        try:
            return spec.coerce_value(raw_val)
        except (ValueError, TypeError) as exc:
            err_str = str(exc)
            choices = spec.choices_tuple
            if choices is not None and "Invalid choice" in err_str:
                bad_item = raw_val
                if isinstance(raw_val, list) and not spec.fixed_tuple_types:
                    for candidate in raw_val:
                        if candidate not in choices and str(candidate) not in {str(c) for c in choices}:
                            bad_item = candidate
                            break
                suggestions = _fuzzy_suggest(str(bad_item), [str(c) for c in choices])
                choices_str = ", ".join(repr(c) for c in choices)
                raise ParseError(
                    f"argument {label}: invalid choice: {bad_item!r} (choose from {choices_str})",
                    usage=usage,
                    suggestions=suggestions,
                    command_path=prog,
                ) from exc

            display_val = raw_val[0] if isinstance(raw_val, list) and len(raw_val) == 1 else raw_val
            raise ParseError(
                f"argument {label}: invalid {spec.type_label} value: {display_val!r}",
                usage=usage,
                command_path=prog,
            ) from exc

    def parse(
        self,
        tokens: Sequence[str],
        *,
        prog: str,
        usage: str = "",
        allow_version: bool = False,
        enable_agent_flags: bool = False,
        allow_mcp: bool = False,
    ) -> ScanResult:
        r"""Scan and coerce leaf CLI tokens into parameter values without intermediate `Namespace` objects.

        Args:
            tokens (Sequence[str]): Raw CLI string tokens for this command level.
            prog (str): Program or command path label used in error messages.
            usage (str): Formatted plain-text usage line attached to any `ParseError`.
            allow_version (bool): Whether `-v` / `--version` is active at this level.
            enable_agent_flags (bool): Whether opt-in agent flags (`--json`, `--dry-run`, etc.) are enabled.
            allow_mcp (bool): Whether `--mcp` is active at this level.

        Returns:
            ScanResult: Resolved parameter dictionary and runtime mode flags.

        Raises:
            ParseError: If unknown flags, invalid values/choices, or missing required arguments are detected.
        """
        flag_map = self.flag_map

        # 1. Fast pre-scan before `--` for eager zero-arg introspection flags and `--json` mode hint
        verbose = False
        json_mode = False
        dry_run = False
        for tok in tokens:
            if tok == "--":
                break
            if enable_agent_flags and tok == "--json" and "--json" not in flag_map:
                json_mode = True
            if tok in ("-h", "--help") and tok not in flag_map:
                return ScanResult(values={}, verbose=verbose, json_mode=json_mode, eager_action="help")
            if allow_version and tok in ("-v", "--version") and tok not in flag_map:
                return ScanResult(values={}, verbose=verbose, json_mode=json_mode, eager_action="version")
            if enable_agent_flags and tok not in flag_map:
                match tok:
                    case "--schema":
                        return ScanResult(values={}, verbose=verbose, json_mode=json_mode, eager_action="schema")
                    case "--help-md":
                        return ScanResult(values={}, verbose=verbose, json_mode=json_mode, eager_action="help_md")
                    case "--skill":
                        return ScanResult(values={}, verbose=verbose, json_mode=json_mode, eager_action="skill")
                    case "--mcp" if allow_mcp:
                        return ScanResult(values={}, verbose=verbose, json_mode=json_mode, eager_action="mcp")

        # 2. Single-pass strict token scan
        raw_opt_tokens: dict[str, Any] = {}
        provided_opts: set[str] = set()
        pos_tokens: list[str] = []
        after_double_dash = False
        idx = 0
        total = len(tokens)

        while idx < total:
            tok = tokens[idx]
            if after_double_dash:
                pos_tokens.append(tok)
                idx += 1
                continue

            if tok == "--":
                after_double_dash = True
                idx += 1
                continue

            if len(tok) > 1 and tok[0] == "-":
                if "=" in tok:
                    flag_part, inline_val = tok.split("=", 1)
                    has_inline = True
                else:
                    flag_part = tok
                    inline_val = ""
                    has_inline = False

                binding = flag_map.get(flag_part)
                if binding is not None:
                    spec = binding.spec
                    provided_opts.add(spec.name)

                    if binding.implicit_bool is not None:
                        if has_inline:
                            if binding.implicit_bool is False:
                                raise ParseError(
                                    f"argument {flag_part}: ignored explicit argument {inline_val!r}",
                                    usage=usage,
                                    command_path=prog,
                                )
                            try:
                                raw_opt_tokens[spec.name] = _parse_bool(inline_val)
                            except ValueError as exc:
                                raise ParseError(
                                    f"argument {flag_part}: invalid boolean value: {inline_val!r}",
                                    usage=usage,
                                    command_path=prog,
                                ) from exc
                        else:
                            raw_opt_tokens[spec.name] = binding.implicit_bool
                        idx += 1
                        continue

                    if spec.is_variadic_collection:
                        bucket: list[str] = raw_opt_tokens.setdefault(spec.name, [])
                        if has_inline:
                            if inline_val:
                                bucket.append(inline_val)
                            elif spec.token_arity[0] >= 1:
                                raise ParseError(
                                    f"argument {flag_part}: expected at least one argument",
                                    usage=usage,
                                    command_path=prog,
                                )
                            idx += 1
                            continue

                        idx += 1
                        consumed_count = 0
                        while idx < total:
                            nxt = tokens[idx]
                            if nxt == "--":
                                break
                            if len(nxt) > 1 and nxt[0] == "-" and not _is_negative_number(nxt):
                                break
                            bucket.append(nxt)
                            consumed_count += 1
                            idx += 1

                        if consumed_count == 0 and spec.token_arity[0] >= 1:
                            raise ParseError(
                                f"argument {flag_part}: expected at least one argument",
                                usage=usage,
                                command_path=prog,
                            )
                        continue

                    fixed = spec.fixed_tuple_types
                    if fixed is not None:
                        arity = len(fixed)
                        if has_inline:
                            if arity != 1:
                                raise ParseError(
                                    f"argument {flag_part}: expected {arity} arguments",
                                    usage=usage,
                                    command_path=prog,
                                )
                            raw_opt_tokens[spec.name] = [inline_val]
                            idx += 1
                            continue

                        idx += 1
                        collected: list[str] = []
                        while len(collected) < arity and idx < total:
                            nxt = tokens[idx]
                            if nxt == "--" or (len(nxt) > 1 and nxt[0] == "-" and not _is_negative_number(nxt)):
                                break
                            collected.append(nxt)
                            idx += 1
                        if len(collected) != arity:
                            raise ParseError(
                                f"argument {flag_part}: expected {arity} arguments",
                                usage=usage,
                                command_path=prog,
                            )
                        raw_opt_tokens[spec.name] = collected
                        continue

                    # Scalar option (arity 1)
                    if has_inline:
                        raw_opt_tokens[spec.name] = inline_val
                        idx += 1
                        continue

                    idx += 1
                    if idx >= total:
                        raise ParseError(
                            f"argument {flag_part}: expected one argument",
                            usage=usage,
                            command_path=prog,
                        )
                    nxt_tok = tokens[idx]
                    if nxt_tok == "--" or (len(nxt_tok) > 1 and nxt_tok[0] == "-" and not _is_negative_number(nxt_tok)):
                        raise ParseError(
                            f"argument {flag_part}: expected one argument",
                            usage=usage,
                            command_path=prog,
                        )
                    raw_opt_tokens[spec.name] = nxt_tok
                    idx += 1
                    continue

                if not has_inline:
                    if flag_part in ("-V", "--verbose"):
                        verbose = True
                        idx += 1
                        continue
                    if enable_agent_flags and flag_part == "--json":
                        json_mode = True
                        idx += 1
                        continue
                    if enable_agent_flags and flag_part == "--dry-run":
                        dry_run = True
                        idx += 1
                        continue

                if _is_negative_number(tok):
                    pos_tokens.append(tok)
                    idx += 1
                    continue

                pool = self._builtin_flag_pool(
                    allow_version=allow_version,
                    enable_agent_flags=enable_agent_flags,
                    allow_mcp=allow_mcp,
                )
                suggestions = _fuzzy_suggest(flag_part, pool)
                raise ParseError(
                    f"unrecognized arguments: {tok}",
                    usage=usage,
                    suggestions=suggestions,
                    command_path=prog,
                )

            pos_tokens.append(tok)
            idx += 1

        # 3. Resolve positional arguments, options, env fallbacks, and defaults
        resolved: dict[str, Any] = {}
        missing_required: list[str] = []

        pos_idx = 0
        num_pos_tokens = len(pos_tokens)

        for spec_i, spec in enumerate(self.positionals):
            if spec.is_variadic_collection:
                min_after = sum(
                    len(rem.fixed_tuple_types) if rem.fixed_tuple_types is not None else (1 if rem.is_required else 0)
                    for rem in self.positionals[spec_i + 1 :]
                )
                end_idx = max(pos_idx, num_pos_tokens - min_after)
                chunk = pos_tokens[pos_idx:end_idx]
                pos_idx = end_idx
                if chunk:
                    resolved[spec.name] = self._coerce_with_parse_error(
                        spec,
                        chunk,
                        label=spec.name,
                        prog=prog,
                        usage=usage,
                    )
                else:
                    meta = spec.arg_meta
                    if meta is not None and meta.env and meta.env in os.environ:
                        resolved[spec.name] = self._coerce_with_parse_error(
                            spec,
                            os.environ[meta.env],
                            label=spec.name,
                            prog=prog,
                            usage=usage,
                        )
                    elif spec.has_default:
                        resolved[spec.name] = _clone_default(spec.default)
                    else:
                        missing_required.append(spec.name)
                continue

            fixed = spec.fixed_tuple_types
            if fixed is not None:
                arity = len(fixed)
                if pos_idx == num_pos_tokens:
                    meta = spec.arg_meta
                    if meta is not None and meta.env and meta.env in os.environ:
                        resolved[spec.name] = self._coerce_with_parse_error(
                            spec,
                            os.environ[meta.env],
                            label=spec.name,
                            prog=prog,
                            usage=usage,
                        )
                    elif spec.has_default:
                        resolved[spec.name] = _clone_default(spec.default)
                    else:
                        missing_required.append(spec.name)
                elif pos_idx + arity > num_pos_tokens:
                    raise ParseError(
                        f"argument {spec.name}: expected {arity} arguments",
                        usage=usage,
                        command_path=prog,
                    )
                else:
                    chunk = pos_tokens[pos_idx : pos_idx + arity]
                    pos_idx += arity
                    resolved[spec.name] = self._coerce_with_parse_error(
                        spec,
                        chunk,
                        label=spec.name,
                        prog=prog,
                        usage=usage,
                    )
                continue

            if pos_idx < num_pos_tokens:
                tok_val = pos_tokens[pos_idx]
                pos_idx += 1
                resolved[spec.name] = self._coerce_with_parse_error(
                    spec,
                    tok_val,
                    label=spec.name,
                    prog=prog,
                    usage=usage,
                )
            else:
                meta = spec.arg_meta
                if meta is not None and meta.env and meta.env in os.environ:
                    resolved[spec.name] = self._coerce_with_parse_error(
                        spec,
                        os.environ[meta.env],
                        label=spec.name,
                        prog=prog,
                        usage=usage,
                    )
                elif spec.has_default:
                    resolved[spec.name] = _clone_default(spec.default)
                else:
                    missing_required.append(spec.name)

        if pos_idx < num_pos_tokens:
            extras = " ".join(pos_tokens[pos_idx:])
            raise ParseError(
                f"unrecognized arguments: {extras}",
                usage=usage,
                command_path=prog,
            )

        for spec in self.options:
            if spec.name in provided_opts:
                resolved[spec.name] = self._coerce_with_parse_error(
                    spec,
                    raw_opt_tokens[spec.name],
                    label=spec.flag_name,
                    prog=prog,
                    usage=usage,
                )
            else:
                meta = spec.arg_meta
                if meta is not None and meta.env and meta.env in os.environ:
                    resolved[spec.name] = self._coerce_with_parse_error(
                        spec,
                        os.environ[meta.env],
                        label=spec.flag_name,
                        prog=prog,
                        usage=usage,
                    )
                elif spec.has_default:
                    resolved[spec.name] = _clone_default(spec.default)
                else:
                    missing_required.append(spec.flag_name)

        if missing_required:
            raise ParseError(
                f"the following arguments are required: {', '.join(missing_required)}",
                usage=usage,
                command_path=prog,
            )

        return ScanResult(
            values=resolved,
            verbose=verbose,
            json_mode=json_mode,
            dry_run=dry_run,
            eager_action=None,
        )


_PANEL_BORDER_STYLE = Style(fg=Colors.CYAN, dim=True)
_PANEL_TITLE_STYLE = Style(fg=Colors.CYAN, bold=True)
_ERROR_BORDER_STYLE = Style(fg=Colors.RED, bold=True)
_ERROR_TITLE_STYLE = Style(fg=Colors.RED, bold=True)


@dataclass(frozen=True, slots=True)
class HelpFormatter:
    r"""Immutable, zero-dependency ANSI help, usage, and error panel renderer for `clio` commands.

    Renders structured multi-column panels (`╭─ Options ───╮`) by default for interactive
    human terminals and automatically degrades to compact unboxed sections when `CLIO_AGENT=1`
    (`agent_mode=True`) or `TERM=dumb`.

    Attributes:
        max_width (int): Maximum panel / line wrap width in terminal columns.
        col_width (int): Minimum target column offset before descriptions begin.
        indent (int): Leading space indentation for unboxed item rows.
        show_defaults (bool): Append `[default: ...]` / `[required]` hints to option descriptions.
        show_env (bool): Append `[env: ...]` hints to parameters backed by `Arg.env`.
        show_types (bool): Display uppercase `TYPE` / choices column inside parameter sections.
        boxed (bool): Wrap help sections and interactive parse errors in Unicode/ASCII panels.
        box_style (BoxStyle): Border glyph specification (`BoxStyle.ROUNDED` by default).
    """

    max_width: int = 100
    col_width: int = 28
    indent: int = 2
    show_defaults: bool = True
    show_env: bool = True
    show_types: bool = True
    boxed: bool = True
    box_style: BoxStyle = BoxStyle.ROUNDED

    def should_box(self, *, agent_mode: bool | None = None) -> bool:
        r"""Return `True` when boxed panel rendering should be active for the current environment."""
        if not self.boxed or agent_mode is True:
            return False
        if agent_mode is None and os.environ.get("CLIO_AGENT", "").strip().lower() in ("1", "true", "yes", "on"):
            return False
        return os.environ.get("TERM") != "dumb"

    def effective_width(self) -> int:
        r"""Resolve flush panel width as `min(terminal_columns, max_width)` (minimum 40 columns)."""
        term_cols = shutil.get_terminal_size(fallback=(self.max_width, 24)).columns
        detected = term_cols if term_cols > 0 else self.max_width
        return max(40, min(detected, self.max_width))

    def format_error_box(
        self,
        message: str,
        *,
        suggestions: Sequence[str] = (),
        title: str = "Error",
    ) -> str:
        r"""Render a red-bordered interactive error panel with optional fuzzy match hints."""
        lines = [message]
        if suggestions:
            styled_first = Colors.info(suggestions[0])
            lines.append(f"💡 Did you mean '{styled_first}'?")
        return Colors.box(
            lines,
            title=title,
            width=self.effective_width(),
            box_style=self.box_style,
            border_style=_ERROR_BORDER_STYLE,
            title_style=_ERROR_TITLE_STYLE,
        )

    def format_usage(
        self,
        prog: str,
        *,
        positionals: Sequence[ParameterSpec] = (),
        has_options: bool = True,
        has_subcommands: bool = False,
    ) -> str:
        r"""Render a styled single-line `usage: ...` synopsis.

        Args:
            prog (str): Program or command path name.
            positionals (Sequence[ParameterSpec]): Positional parameter specs in order.
            has_options (bool): Whether any option flags exist at this level.
            has_subcommands (bool): Whether subcommands exist under this command/CLI.

        Returns:
            str: Formatted usage line ending in a newline.
        """
        parts: list[str] = [f"{Colors.info('usage: ')}{Colors.bold(prog)}"]
        if has_options:
            parts.append(Colors.info("[options]"))

        for pos in positionals:
            mv = pos.metavar_label
            if pos.is_variadic_collection:
                parts.append(f"{mv} [{mv} ...]" if pos.is_required else f"[{mv} ...]")
            else:
                parts.append(mv if pos.is_required else f"[{mv}]")

        if has_subcommands:
            parts.append(Colors.info("[subcommand]"))

        return " ".join(parts) + "\n"

    def _describe_spec(self, spec: ParameterSpec, *, is_option: bool) -> str:
        chunks: list[str] = []
        if spec.docstring:
            chunks.append(spec.docstring)

        meta = spec.arg_meta
        if self.show_env and meta is not None and meta.env:
            env_tag = f"[env: {meta.env}]"
            if env_tag not in spec.docstring:
                chunks.append(Colors.muted(env_tag))

        if is_option and self.show_defaults:
            if spec.is_required:
                chunks.append(Colors.warning("[required]"))
            elif spec.has_default and spec.default is not None and spec.default != "" and spec.default != []:
                default_val = spec.default.value if isinstance(spec.default, enum.Enum) else spec.default
                chunks.append(Colors.muted(f"[default: {default_val}]"))

        return " ".join(chunks)

    def _spec_to_row(self, spec: ParameterSpec) -> tuple[str, str, str, str, str]:
        r"""Convert a `ParameterSpec` into `(req, primary, short, type_badge, description)` columns."""
        req_cell = Colors.error("*") if spec.is_required else " "
        if spec.is_positional:
            primary_cell = Colors.cyan(spec.name, bold=True)
            short_cell = ""
        else:
            flag_list = list(spec.option_flags)
            if spec.show_negated_in_help and spec.negated_flag is not None and spec.negated_flag not in flag_list:
                flag_list.append(spec.negated_flag)
            long_flags = [f for f in flag_list if f.startswith("--")]
            short_flags = [f for f in flag_list if not f.startswith("--")]
            primary_cell = ", ".join(Colors.cyan(f, bold=True) for f in long_flags)
            short_cell = ", ".join(Colors.green(f, bold=True) for f in short_flags)

        badge = spec.badge_label if self.show_types else ""
        type_cell = Colors.yellow(badge) if badge else ""
        desc = self._describe_spec(spec, is_option=not spec.is_positional)
        return (req_cell, primary_cell, short_cell, type_cell, desc)

    @staticmethod
    def _flag_pair_to_row(flags: Sequence[str], desc: str) -> tuple[str, str, str, str, str]:
        r"""Convert built-in global/agent flag strings and description into a 5-column row."""
        long_flags = [f for f in flags if f.startswith("--")]
        short_flags = [f for f in flags if not f.startswith("--")]
        primary_cell = ", ".join(Colors.cyan(f, bold=True) for f in long_flags)
        short_cell = ", ".join(Colors.green(f, bold=True) for f in short_flags)
        return (" ", primary_cell, short_cell, "", desc)

    def _render_table_lines(
        self,
        rows: Sequence[tuple[str, str, str, str, str]],
        *,
        avail_width: int,
        lead_indent: int,
    ) -> list[str]:
        r"""Format 5-column rows into aligned, ANSI-wrapped terminal lines within `avail_width`."""
        has_req = any(bool(r[0]) for r in rows)
        max_prim_cap = max(16, avail_width // 2 - 6)
        prim_widths = [Colors.display_width(r[1]) for r in rows]
        fitting_prims = [w for w in prim_widths if w <= max_prim_cap]
        prim_w = max(fitting_prims) if fitting_prims else min(max_prim_cap, max(prim_widths, default=0))
        short_w = max((Colors.display_width(r[2]) for r in rows), default=0)
        type_w = max((Colors.display_width(r[3]) for r in rows), default=0)

        lead = " " * lead_indent
        base_prefix_w = (
            lead_indent
            + (3 if has_req else 0)
            + prim_w
            + ((2 + short_w) if short_w else 0)
            + ((2 + type_w) if type_w else 0)
        )
        min_col = min(self.col_width, max(16, avail_width // 2))
        desc_col = min(max(base_prefix_w + 2, min_col), max(16, avail_width - 20))
        desc_w = max(18, avail_width - desc_col)
        cont_pad = " " * desc_col

        out: list[str] = []
        for req_c, prim_c, short_c, type_c, desc_c in rows:
            p_vis = Colors.display_width(prim_c)
            s_vis = Colors.display_width(short_c)
            t_vis = Colors.display_width(type_c)
            req_part = f"{req_c}  " if has_req else ""

            if p_vis <= prim_w:
                parts = [f"{lead}{req_part}{prim_c}{' ' * (prim_w - p_vis)}"]
                if short_w:
                    parts.append(f"  {short_c}{' ' * (short_w - s_vis)}")
                if type_w:
                    parts.append(f"  {type_c}{' ' * (type_w - t_vis)}")
                prefix = "".join(parts)
                prefix_vis = base_prefix_w
            else:
                compact = "  ".join(c for c in (prim_c, short_c, type_c) if c)
                prefix = f"{lead}{req_part}{compact}"
                prefix_vis = Colors.display_width(prefix)

            if not desc_c:
                out.append(prefix.rstrip())
                continue

            wrapped = Colors.wrap_ansi(desc_c, desc_w)
            if prefix_vis <= desc_col - 2:
                gap = " " * (desc_col - prefix_vis)
                out.append(f"{prefix}{gap}{wrapped[0]}")
                for cont in wrapped[1:]:
                    out.append(f"{cont_pad}{cont}")
            else:
                out.append(prefix.rstrip())
                for cont in wrapped:
                    out.append(f"{cont_pad}{cont}")

        return out

    def _render_section(
        self,
        title: str,
        rows: Sequence[tuple[str, str, str, str, str]],
        *,
        boxed: bool,
        outer_width: int,
    ) -> str:
        r"""Render a single help section either inside a `Colors.box` panel or as an indented section."""
        if boxed:
            lines = self._render_table_lines(rows, avail_width=max(24, outer_width - 4), lead_indent=0)
            return Colors.box(
                lines,
                title=title,
                width=outer_width,
                padding=1,
                box_style=self.box_style,
                border_style=_PANEL_BORDER_STYLE,
                title_style=_PANEL_TITLE_STYLE,
            )

        lines = self._render_table_lines(rows, avail_width=outer_width, lead_indent=self.indent)
        heading = Colors.info(Colors.bold(f"{title}:"))
        return "\n".join((heading, *lines))

    def format_help(
        self,
        prog: str,
        *,
        description: str | None = None,
        epilog: str | None = None,
        scan_plan: ScanPlan | None = None,
        subcommands: Sequence[tuple[str, Sequence[str], str]] = (),
        subcommands_title: str = "💬 Commands",
        has_version: bool = False,
        enable_agent_flags: bool = False,
        allow_mcp: bool = False,
        agent_mode: bool | None = None,
    ) -> str:
        r"""Render complete multi-section ANSI help output for a root `Cli`, `CommandGroup`, or `Command`.

        Args:
            prog (str): Program or command path name.
            description (str | None): Header description block.
            epilog (str | None): Footer epilog block.
            scan_plan (ScanPlan | None): Compiled `ScanPlan` for this command level, if any.
            subcommands (Sequence[tuple[str, Sequence[str], str]]): Tuple of `(name, aliases, summary)` rows.
            subcommands_title (str): Section heading for subcommands.
            has_version (bool): Whether `-v` / `--version` is registered.
            enable_agent_flags (bool): Whether `🤖 Agent options` are registered.
            allow_mcp (bool): Whether `--mcp` is included in `🤖 Agent options`.
            agent_mode (bool): Explicit agent-mode flag overriding environment detection when `True`.

        Returns:
            str: Complete formatted help text ending with a newline.
        """
        positionals = scan_plan.positionals if scan_plan is not None else ()
        options = scan_plan.options if scan_plan is not None else ()
        shadowed = scan_plan.flag_map if scan_plan is not None else {}

        use_boxes = self.should_box(agent_mode=agent_mode)
        outer_width = self.effective_width()

        usage = self.format_usage(
            prog,
            positionals=positionals,
            has_options=True,
            has_subcommands=bool(subcommands),
        )
        header_blocks: list[str] = [usage.rstrip("\n")]
        if description and description.strip():
            header_blocks.append(Colors.bold(description.strip()))

        panels: list[str] = []

        ungrouped_pos = [p for p in positionals if p.group is None]
        if ungrouped_pos:
            panels.append(
                self._render_section(
                    "Positional arguments",
                    [self._spec_to_row(p) for p in ungrouped_pos],
                    boxed=use_boxes,
                    outer_width=outer_width,
                )
            )

        ungrouped_opts = [o for o in options if o.group is None]
        if ungrouped_opts:
            panels.append(
                self._render_section(
                    "Options",
                    [self._spec_to_row(o) for o in ungrouped_opts],
                    boxed=use_boxes,
                    outer_width=outer_width,
                )
            )

        grouped: dict[str, list[ParameterSpec]] = {}
        for spec in (*positionals, *options):
            if spec.group is not None:
                grouped.setdefault(spec.group, []).append(spec)

        for group_name, group_specs in grouped.items():
            panels.append(
                self._render_section(
                    group_name,
                    [self._spec_to_row(s) for s in group_specs],
                    boxed=use_boxes,
                    outer_width=outer_width,
                )
            )

        if subcommands:
            sub_rows = [
                (
                    "",
                    Colors.cyan(
                        f"{sub_name} ({', '.join(sub_aliases)})" if sub_aliases else sub_name,
                        bold=True,
                    ),
                    "",
                    "",
                    sub_summary,
                )
                for sub_name, sub_aliases, sub_summary in subcommands
            ]
            panels.append(
                self._render_section(
                    subcommands_title,
                    sub_rows,
                    boxed=use_boxes,
                    outer_width=outer_width,
                )
            )

        global_rows: list[tuple[str, str, str, str, str]] = []
        help_flags = [f for f in ("-h", "--help") if f not in shadowed]
        if help_flags:
            global_rows.append(self._flag_pair_to_row(help_flags, "Show this help message and exit"))
        verbose_flags = [f for f in ("-V", "--verbose") if f not in shadowed]
        if verbose_flags:
            global_rows.append(self._flag_pair_to_row(verbose_flags, "Enable verbose output/progress"))
        if has_version:
            ver_flags = [f for f in ("-v", "--version") if f not in shadowed]
            if ver_flags:
                global_rows.append(self._flag_pair_to_row(ver_flags, "Show program's version number and exit"))

        if global_rows:
            panels.append(
                self._render_section(
                    "🌎 Global options",
                    global_rows,
                    boxed=use_boxes,
                    outer_width=outer_width,
                )
            )

        if enable_agent_flags:
            agent_candidates: list[tuple[str, str]] = [
                ("--json", "Emit structured JSON execution envelope"),
                ("--dry-run", "Validate arguments and inspect resolved CallPlan without executing"),
                ("--schema", "Print JSON Schema specification and exit"),
                ("--help-md", "Print Markdown reference documentation and exit"),
                ("--skill", "Print Agent SKILL.md specification and exit"),
            ]
            if allow_mcp:
                agent_candidates.append(("--mcp", "Start stdio Model Context Protocol (MCP) JSON-RPC server"))

            agent_rows = [
                self._flag_pair_to_row((flag,), desc) for flag, desc in agent_candidates if flag not in shadowed
            ]
            if agent_rows:
                panels.append(
                    self._render_section(
                        "🤖 Agent options",
                        agent_rows,
                        boxed=use_boxes,
                        outer_width=outer_width,
                    )
                )

        body_blocks: list[str] = ["\n\n".join(header_blocks)]
        if panels:
            panel_sep = "\n" if use_boxes else "\n\n"
            body_blocks.append(panel_sep.join(panels))
        if epilog and epilog.strip():
            body_blocks.append(Colors.bold(epilog.strip()))

        return "\n\n".join(body_blocks) + "\n"


_DEFAULT_FORMATTER = HelpFormatter()
