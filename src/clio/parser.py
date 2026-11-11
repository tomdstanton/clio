import enum
import os
import textwrap
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

from clio.color import Colors
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


@dataclass(frozen=True, slots=True)
class HelpFormatter:
    r"""Immutable, zero-dependency ANSI help and usage renderer for `clio` commands.

    Attributes:
        max_width (int): Maximum line wrap width in columns.
        col_width (int): Width reserved for the left-hand argument/flag column.
        indent (int): Leading space indentation for item rows.
        show_defaults (bool): Append `[default: ...]` / `[required]` hints to option descriptions.
        show_env (bool): Append `[env: ...]` hints to parameters backed by `Arg.env`.
    """

    max_width: int = 100
    col_width: int = 28
    indent: int = 2
    show_defaults: bool = True
    show_env: bool = True

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
        parts: list[str] = [f"{Colors.info('usage: ')}{prog}"]
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

    def _format_row(self, left: str, right: str) -> str:
        pad = " " * self.indent
        vis_left = Colors.visible_len(left)
        target_col = max(self.col_width, self.indent + 4)
        inner_col = target_col - self.indent

        if not right:
            return f"{pad}{left}"

        wrap_width = max(30, self.max_width - target_col)
        raw_paragraphs = right.splitlines() or [""]
        wrapped_lines: list[str] = []
        for para in raw_paragraphs:
            if not para.strip():
                wrapped_lines.append("")
                continue
            # Preserve ANSI sequences by wrapping when visible length exceeds wrap_width
            if Colors.visible_len(para) <= wrap_width:
                wrapped_lines.append(para)
            else:
                wrapped_lines.extend(textwrap.wrap(para, width=wrap_width) or [para])

        cont_indent = " " * target_col
        if vis_left <= inner_col - 2:
            gap = " " * (inner_col - vis_left)
            first = f"{pad}{left}{gap}{wrapped_lines[0]}"
            if len(wrapped_lines) == 1:
                return first
            rest = "\n".join(f"{cont_indent}{line}" for line in wrapped_lines[1:])
            return f"{first}\n{rest}"

        body = "\n".join(f"{cont_indent}{line}" for line in wrapped_lines)
        return f"{pad}{left}\n{body}"

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

        Returns:
            str: Complete formatted help text ending with a newline.
        """
        positionals = scan_plan.positionals if scan_plan is not None else ()
        options = scan_plan.options if scan_plan is not None else ()
        shadowed = scan_plan.flag_map if scan_plan is not None else {}

        usage = self.format_usage(
            prog,
            positionals=positionals,
            has_options=True,
            has_subcommands=bool(subcommands),
        )
        sections: list[str] = [usage.rstrip("\n")]

        if description and description.strip():
            sections.append(Colors.bold(description.strip()))

        if positionals:
            pos_lines = [Colors.info("Positional arguments:")]
            for pos in positionals:
                choices = pos.choices_tuple
                left = f"{pos.name} {pos.metavar_label}" if choices is not None else pos.name
                pos_lines.append(self._format_row(left, self._describe_spec(pos, is_option=False)))
            sections.append("\n".join(pos_lines))

        if options:
            opt_lines = [Colors.info("Options:")]
            for opt in options:
                flag_list = list(opt.option_flags)
                if opt.annotation is bool:
                    if opt.show_negated_in_help and opt.negated_flag is not None and opt.negated_flag not in flag_list:
                        flag_list.append(opt.negated_flag)
                    left = ", ".join(flag_list)
                else:
                    left = f"{', '.join(flag_list)} {opt.metavar_label}"
                opt_lines.append(self._format_row(left, self._describe_spec(opt, is_option=True)))
            sections.append("\n".join(opt_lines))

        if subcommands:
            sub_heading = Colors.info(Colors.bold(f"{subcommands_title}:"))
            sub_lines = [sub_heading]
            for sub_name, sub_aliases, sub_summary in subcommands:
                label = f"{sub_name} ({', '.join(sub_aliases)})" if sub_aliases else sub_name
                sub_lines.append(self._format_row(label, sub_summary))
            sections.append("\n".join(sub_lines))

        global_rows: list[tuple[str, str]] = []
        help_flags = [f for f in ("-h", "--help") if f not in shadowed]
        if help_flags:
            global_rows.append((", ".join(help_flags), "Show this help message and exit"))
        verbose_flags = [f for f in ("-V", "--verbose") if f not in shadowed]
        if verbose_flags:
            global_rows.append((", ".join(verbose_flags), "Enable verbose output/progress"))
        if has_version:
            ver_flags = [f for f in ("-v", "--version") if f not in shadowed]
            if ver_flags:
                global_rows.append((", ".join(ver_flags), "Show program's version number and exit"))

        if global_rows:
            g_heading = Colors.info(Colors.bold("🌎 Global options:"))
            g_lines = [g_heading, *(self._format_row(left, right) for left, right in global_rows)]
            sections.append("\n".join(g_lines))

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

            agent_rows = [(flag, desc) for flag, desc in agent_candidates if flag not in shadowed]
            if agent_rows:
                a_heading = Colors.info(Colors.bold("🤖 Agent options:"))
                a_lines = [a_heading, *(self._format_row(left, right) for left, right in agent_rows)]
                sections.append("\n".join(a_lines))

        if epilog and epilog.strip():
            sections.append(Colors.bold(epilog.strip()))

        return "\n\n".join(sections) + "\n"


_DEFAULT_FORMATTER = HelpFormatter()
