import argparse
import inspect
import json
import os
import re
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, Any, ClassVar, Self, TextIO

from clio.agent import (
    AgentErrorSpec,
    AgentMessage,
    AgentResult,
    McpServer,
    ToolSpec,
    _detect_agent_mode,
    _OutputCaptureContext,
)
from clio.color import ColorHelpFormatter, Colors
from clio.command import BaseCommand, _coerce_subcommand, _escape_md_cell, _format_action_default, _format_action_type
from clio.docstr import DocstringParser


class CliError(Exception):
    r"""Base exception for expected CLI application errors.

    Exceptions inheriting from `CliError` (or passed via `catch_exceptions` to [`Cli`][clio.Cli])
    are caught gracefully by the runner, printed to `sys.stderr` with formatting, and exit with
    a non-zero status code without dumping a raw Python traceback.
    """

    __slots__ = ("exit_code",)

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class PackageMeta:
    r"""Immutable package distribution metadata extracted via `importlib.metadata`.

    Attributes:
        name (str): Distribution package name.
        version (str): Installed distribution version string.
        summary (str): One-line package summary from `pyproject.toml` / `METADATA`.
        author (str): Package author name or email contact.
        license (str): Package license identifier.
        homepage (str): Homepage URL if declared in distribution metadata.
        repository (str): Source repository URL if declared in distribution metadata.
        documentation (str): Documentation URL if declared in distribution metadata.
        urls (Mapping[str, str]): Read-only mapping of all `Project-URL` entries.
    """

    _CACHE: ClassVar[dict[str, PackageMeta | None]] = {}

    name: str = ""
    version: str = ""
    summary: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    repository: str = ""
    documentation: str = ""
    urls: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(cls, package: str) -> PackageMeta | None:
        r"""Resolve and cache distribution metadata for an installed Python package.

        Args:
            package (str): Distribution or top-level module name to query.

        Returns:
            PackageMeta | None: Frozen `PackageMeta` instance, or `None` if package is not installed.
        """
        if not package:
            return None
        if package in cls._CACHE:
            return cls._CACHE[package]

        import importlib.metadata

        try:
            meta = importlib.metadata.metadata(package)
        except importlib.metadata.PackageNotFoundError:
            cls._CACHE[package] = None
            return None

        urls: dict[str, str] = {}
        for raw_url in meta.get_all("Project-URL") or ():
            if "," in raw_url:
                label, url = raw_url.split(",", 1)
                urls[label.strip()] = url.strip()

        lower_urls = {k.lower(): v for k, v in urls.items()}
        homepage = lower_urls.get("homepage") or meta.get("Home-page") or ""
        repository = lower_urls.get("repository") or lower_urls.get("source") or ""
        documentation = lower_urls.get("documentation") or lower_urls.get("docs") or ""

        result = cls(
            name=meta.get("Name") or package,
            version=meta.get("Version") or "",
            summary=meta.get("Summary") or "",
            author=meta.get("Author") or meta.get("Author-email") or "",
            license=meta.get("License-Expression") or meta.get("License") or "",
            homepage=homepage,
            repository=repository,
            documentation=documentation,
            urls=types.MappingProxyType(urls),
        )
        cls._CACHE[package] = result
        return result


@dataclass(frozen=True, slots=True)
class CliConfig:
    r"""Immutable configuration specification for a [`Cli`][clio.cli.Cli] application.

    Attributes:
        prog (str | None): Program name displayed in usage.
        description (str | None): Header description text displayed in top-level help.
        epilog (str | None): Footer epilog text displayed in top-level help.
        version (str | None): Program version string.
        catch_exceptions (tuple[type[Exception], ...]): Exception classes trapped cleanly without traceback.
        formatter_class (type[argparse.HelpFormatter]): Help formatter class applied across parsers.
        package_meta (PackageMeta | None): Resolved `importlib.metadata` package specification, if any.
        enable_agent_flags (bool): Whether opt-in agent CLI flags (`--json`, `--dry-run`, `--schema`,
            `--help-md`, `--skill`, `--mcp`) are registered on parsers.
    """

    prog: str | None = None
    description: str | None = None
    epilog: str | None = None
    version: str | None = None
    catch_exceptions: tuple[type[Exception], ...] = (CliError,)
    formatter_class: type[argparse.HelpFormatter] = ColorHelpFormatter
    package_meta: PackageMeta | None = None
    enable_agent_flags: bool = False


class _AgentIntrospectAction(argparse.Action):
    r"""Eager zero-argument `argparse.Action` for `--schema`, `--help-md`, `--skill`, and `--mcp`."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = argparse.SUPPRESS,
        *,
        cli: Cli,
        mode: str,
        command: BaseCommand | None = None,
        help: str | None = None,
    ) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=argparse.SUPPRESS,
            nargs=0,
            help=help,
        )
        self._cli = cli
        self._mode = mode
        self._command = command

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        if self._mode == "schema":
            payload = self._command.to_json_schema() if self._command is not None else self._cli.to_json_schema()
            print(json.dumps(payload, indent=2))
        elif self._mode == "help_md":
            md = (
                self._command.to_markdown(prog_prefix=self._cli.parser.prog)
                if self._command is not None
                else self._cli.to_markdown()
            )
            print(md)
        elif self._mode == "skill":
            skill = (
                self._command.to_skill_md(
                    prog_prefix=self._cli.parser.prog,
                    enable_agent_flags=self._cli.config.enable_agent_flags,
                )
                if self._command is not None
                else self._cli.to_skill_md()
            )
            print(skill)
        elif self._mode == "mcp":
            self._cli.serve_mcp()
        parser.exit(0)


class HelpOnErrorParser(argparse.ArgumentParser):
    r"""Argument parser that outputs full help text and suggestions on error.

    Extends `argparse.ArgumentParser` to print help menu to
    `sys.stderr` when invalid arguments or choices are encountered, suggesting close matches.
    Supports token-efficient error formatting when `CLIO_AGENT=1` is active and structured
    JSON error envelopes when `--json` is enabled.
    """

    _INVALID_CHOICE_RE: ClassVar[re.Pattern[str]] = re.compile(r"invalid choice: '?([^']+)'? \(choose from (.*)\)")
    _UNRECOGNIZED_ARGS_RE: ClassVar[re.Pattern[str]] = re.compile(r"unrecognized arguments:\s*(.+)")
    _ACTIVE_JSON_MODE: ClassVar[bool] = False
    _ACTIVE_AGENT_MODE: ClassVar[bool | None] = None
    _LAST_SUBPARSER: ClassVar[HelpOnErrorParser | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def parse_known_args(  # ty: ignore[invalid-method-override]
        self,
        args: Sequence[str] | None = None,
        namespace: Any = None,
    ) -> tuple[Any, list[str]]:
        r"""Track the deepest active subparser during argument parsing."""
        HelpOnErrorParser._LAST_SUBPARSER = self
        return super().parse_known_args(args, namespace)

    def error(self, message: str) -> None:  # type: ignore
        r"""Print help text and error message before terminating CLI session.

        Args:
            message (str): Error message string reported by parser.

        Raises:
            SystemExit: Exits process with status code 2.
        """
        if (
            HelpOnErrorParser._LAST_SUBPARSER is not None
            and HelpOnErrorParser._LAST_SUBPARSER is not self
            and self._UNRECOGNIZED_ARGS_RE.search(message)
        ):
            target = HelpOnErrorParser._LAST_SUBPARSER
            HelpOnErrorParser._LAST_SUBPARSER = None
            target.error(message)
            return

        from difflib import get_close_matches

        suggestions: list[str] = []
        display_message = message

        if match := self._INVALID_CHOICE_RE.search(message):
            invalid = match.group(1)
            choices = [c.strip("'").strip() for c in match.group(2).split(", ")]

            if matches := get_close_matches(invalid, choices):
                suggestions.extend(matches)
                display_message += f"\n    💡 Did you mean '{Colors.info(matches[0])}'?"
        elif match := self._UNRECOGNIZED_ARGS_RE.search(message):
            raw_tokens = match.group(1).strip().split()
            option_keys = list(self._option_string_actions.keys())
            for tok in raw_tokens:
                flag_candidate = tok.split("=", 1)[0]
                if flag_candidate.startswith("-") and option_keys:
                    if matches := get_close_matches(flag_candidate, option_keys):
                        suggestions.extend(matches)
                        display_message += f"\n    💡 Did you mean '{Colors.info(matches[0])}'?"
                        break

        if self._ACTIVE_JSON_MODE:
            clean_usage = Colors.strip_ansi(self.format_usage()).strip()
            clean_msg = Colors.strip_ansi(message).strip()
            err_result = AgentResult(
                ok=False,
                command=self.prog,
                error=AgentErrorSpec(
                    message=clean_msg,
                    exit_code=2,
                    usage=clean_usage,
                    suggestions=tuple(suggestions),
                ),
            )
            print(err_result.to_json())
            self.exit(2)

        is_agent = self._ACTIVE_AGENT_MODE if self._ACTIVE_AGENT_MODE is not None else _detect_agent_mode()
        if is_agent:
            usage_line = Colors.strip_ansi(self.format_usage()).strip()
            hint_line = f"    Run '{self.prog} --help' for full help.\n"
            self.exit(2, f"{usage_line}\n{Colors.error('❌ Error:')} {display_message}\n{hint_line}")

        self.print_help(sys.stderr)
        self.exit(2, f"\n{Colors.error('❌ Error:')} {display_message}\n")


class Cli:
    r"""Root Command Line Interface runner and context manager.

    Coordinates global option parsing, subcommand registration, safe file handle
    tracking, error trapping, and progress updates.

    Attributes:
        config (CliConfig): Immutable configuration specification for this CLI instance.
        verbose (bool): Flag indicating whether verbose logging is enabled.
        catch_exceptions (tuple[type[Exception], ...]): Exception types to catch gracefully.
        global_parser (HelpOnErrorParser): Parser handling shared global options.
        parser (HelpOnErrorParser): Main argument parser for application commands.
        subparsers (argparse._SubParsersAction): Subparser registry for subcommands.
    """

    def __init__(
        self,
        prog: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        version: str | None = None,
        catch_exceptions: Iterable[type[Exception]] = (),
        formatter_class: type[argparse.HelpFormatter] = ColorHelpFormatter,
        commands: Iterable[BaseCommand | type[Any]] = (),
        package: str | None = None,
        *,
        enable_agent_flags: bool = False,
        agent_mode: bool | None = None,
    ) -> None:
        r"""Initialize root CLI application instance.

        Automatically queries `importlib.metadata` when `package` is provided (or when `prog` or the
        calling package matches an installed distribution) to populate omitted `prog`, `description`,
        and `version` fields.

        Args:
            prog (str | None): Program name displayed in usage (defaults to package name or `sys.argv[0]`).
            description (str | None): Header description text displayed in top-level help.
            epilog (str | None): Footer epilog text displayed in top-level help.
            version (str | None): Program version string. If provided (or resolved via `importlib.metadata`),
                registers `-v`/`--version`.
            catch_exceptions (Iterable[type[Exception]]): Domain exception classes to trap and
                format cleanly to `sys.stderr` without a traceback. `CliError` is always included.
            formatter_class (type[argparse.HelpFormatter]): Help formatter class to apply across
                root and subcommand parsers. Defaults to [`ColorHelpFormatter`][clio.ColorHelpFormatter].
            commands (Iterable[BaseCommand | type[Any]]): Optional sequence of commands (classes,
                dataclasses, or instances) to register immediately.
            package (str | None): Optional distribution package name to query via `importlib.metadata`.
            enable_agent_flags (bool): Opt in to registering built-in agent CLI flags (`--json`,
                `--dry-run`, `--schema`, `--help-md`, `--skill`, `--mcp`) in an `Agent Options` group.
            agent_mode (bool | None): Override `CLIO_AGENT=1` environment detection for concise
                error output and disabled `\r` spinners.
        """
        candidate_pkg = package
        if candidate_pkg is None and prog is None:
            if type(self) is not Cli:
                mod_root = type(self).__module__.split(".", 1)[0]
                if mod_root not in ("__main__", "builtins", "clio"):
                    candidate_pkg = mod_root
            if candidate_pkg is None and (frame := inspect.currentframe()) is not None and frame.f_back is not None:
                caller_pkg = frame.f_back.f_globals.get("__package__")
                if isinstance(caller_pkg, str) and caller_pkg:
                    candidate_pkg = caller_pkg.split(".", 1)[0]

        pkg_meta = PackageMeta.resolve(candidate_pkg) if candidate_pkg else None

        if description is None and type(self) is not Cli:
            parsed_desc, _ = DocstringParser.parse_description(type(self))
            if parsed_desc:
                description = parsed_desc

        if pkg_meta is not None:
            if prog is None and pkg_meta.name:
                prog = pkg_meta.name
            if description is None and pkg_meta.summary:
                description = pkg_meta.summary
            if version is None and pkg_meta.version:
                version = pkg_meta.version

        self.config = CliConfig(
            prog=prog,
            description=description,
            epilog=epilog,
            version=version,
            catch_exceptions=(CliError, *catch_exceptions),
            formatter_class=formatter_class,
            package_meta=pkg_meta,
            enable_agent_flags=enable_agent_flags,
        )
        self.verbose = False
        self._agent_mode_override = agent_mode
        self._active_message_sink: list[AgentMessage] | None = None
        self.formatter_class = self.config.formatter_class
        self.catch_exceptions: tuple[type[Exception], ...] = self.config.catch_exceptions

        self.global_parser = HelpOnErrorParser(add_help=False)
        self.global_parser.add_argument(
            "-V",
            "--verbose",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable verbose output/progress",
        )

        self.parser = HelpOnErrorParser(
            prog=self.config.prog,
            description=Colors.bold(description) if description else description,
            epilog=Colors.bold(epilog) if epilog else epilog,
            parents=[self.global_parser],
            formatter_class=self.formatter_class,
        )

        if version is not None:
            self.parser.add_argument(
                "-v",
                "--version",
                action="version",
                version=f"%(prog)s {version}",
                help="Show program's version number and exit",
            )

        # Rename the default options group for the root parser
        if hasattr(self.parser, "_optionals"):
            self.parser._optionals.title = Colors.bold("🌎 Global options")

        if self.config.enable_agent_flags:
            self._attach_agent_options(self.parser, command=None)

        self.subparsers = self.parser.add_subparsers(title=Colors.bold("💬 Commands"), dest="command", required=True)
        self.commands: list[BaseCommand] = []
        self._open_handles: list[IO[Any]] = []
        self.add_commands(*commands)

    @property
    def agent_mode(self) -> bool:
        r"""Whether token-efficient agent runtime mode (`CLIO_AGENT=1`) is active."""
        if self._agent_mode_override is not None:
            return self._agent_mode_override
        return _detect_agent_mode()

    def _attach_agent_options(
        self,
        parser: argparse.ArgumentParser,
        *,
        command: BaseCommand | None = None,
    ) -> None:
        r"""Attach the opt-in `🤖 Agent options` group onto `parser` without colliding with user flags."""
        existing_opts = set(parser._option_string_actions.keys())
        group = parser.add_argument_group(Colors.bold("🤖 Agent options"))
        is_root = command is None
        flag_default: Any = False if is_root else argparse.SUPPRESS

        if "--json" not in existing_opts:
            group.add_argument(
                "--json",
                dest="_clio_json",
                action="store_true",
                default=flag_default,
                help="Emit structured JSON execution envelope",
            )
        if "--dry-run" not in existing_opts:
            group.add_argument(
                "--dry-run",
                dest="_clio_dry_run",
                action="store_true",
                default=flag_default,
                help="Validate arguments and inspect resolved CallPlan without executing",
            )
        if "--schema" not in existing_opts:
            group.add_argument(
                "--schema",
                action=_AgentIntrospectAction,
                cli=self,
                mode="schema",
                command=command,
                help="Print JSON Schema specification and exit",
            )
        if "--help-md" not in existing_opts:
            group.add_argument(
                "--help-md",
                action=_AgentIntrospectAction,
                cli=self,
                mode="help_md",
                command=command,
                help="Print Markdown reference documentation and exit",
            )
        if "--skill" not in existing_opts:
            group.add_argument(
                "--skill",
                action=_AgentIntrospectAction,
                cli=self,
                mode="skill",
                command=command,
                help="Print Agent SKILL.md specification and exit",
            )
        if is_root and "--mcp" not in existing_opts:
            group.add_argument(
                "--mcp",
                action=_AgentIntrospectAction,
                cli=self,
                mode="mcp",
                command=None,
                help="Start stdio Model Context Protocol (MCP) JSON-RPC server",
            )

    @classmethod
    def from_package(
        cls,
        package: str,
        *,
        prog: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        version: str | None = None,
        catch_exceptions: Iterable[type[Exception]] = (),
        formatter_class: type[argparse.HelpFormatter] = ColorHelpFormatter,
        commands: Iterable[BaseCommand | type[Any]] = (),
        enable_agent_flags: bool = False,
        agent_mode: bool | None = None,
    ) -> Self:
        r"""Construct a `Cli` instance pre-populated from an installed package's `importlib.metadata`.

        Args:
            package (str): Installed Python distribution name.
            prog (str | None): Optional override for program name.
            description (str | None): Optional override for header description.
            epilog (str | None): Optional footer epilog text.
            version (str | None): Optional override for version string.
            catch_exceptions (Iterable[type[Exception]]): Exception types to trap cleanly.
            formatter_class (type[argparse.HelpFormatter]): Help formatter class.
            commands (Iterable[BaseCommand | type[Any]]): Commands to register immediately.
            enable_agent_flags (bool): Opt in to built-in agent CLI flags (`--json`, `--dry-run`, etc.).
            agent_mode (bool | None): Override `CLIO_AGENT=1` environment detection.

        Returns:
            Self: Configured `Cli` instance.
        """
        return cls(
            prog=prog,
            description=description,
            epilog=epilog,
            version=version,
            catch_exceptions=catch_exceptions,
            formatter_class=formatter_class,
            commands=commands,
            package=package,
            enable_agent_flags=enable_agent_flags,
            agent_mode=agent_mode,
        )

    def add_command(self, command: BaseCommand | type[Any]) -> Self:
        r"""Build and attach a top-level command (instance, class, or `@dataclass`) to root CLI subparsers.

        Args:
            command (BaseCommand | type[Any]): Subcommand instance, `BaseCommand` subclass, or `@dataclass`.

        Returns:
            Self: The CLI instance for method chaining.
        """
        cmd_instance = _coerce_subcommand(command)
        cmd_instance.cli = self
        cmd_instance.build(
            self.subparsers,
            parent_parsers=[self.global_parser],
            formatter_class=self.formatter_class,
        )
        self.commands.append(cmd_instance)
        return self

    def add_commands(self, *commands: BaseCommand | type[Any]) -> Self:
        r"""Build and attach multiple top-level commands (instances, classes, or `@dataclass` types).

        Args:
            *commands (BaseCommand | type[Any]): Subcommand classes, dataclasses, or instances to register.

        Returns:
            Self: The CLI instance for method chaining.
        """
        for command in commands:
            self.add_command(command)
        return self

    def to_markdown(
        self,
        *,
        heading_level: int = 1,
        include_global_options: bool = False,
    ) -> str:
        r"""Generate zero-dependency GitHub-flavored Markdown documentation for the entire CLI application.

        Args:
            heading_level (int): Root Markdown heading depth (`1` for `#`, `2` for `##`, etc.).
            include_global_options (bool): Whether to repeat global options (`-h`, `-V`) on every subcommand table.

        Returns:
            str: Complete Markdown reference documentation for the CLI and all registered commands.
        """
        prog = self.parser.prog
        hashes = "#" * max(1, heading_level)
        lines: list[str] = [f"{hashes} `{prog}`"]

        if self.config.description:
            lines.append(f"\n{Colors.strip_ansi(self.config.description).strip()}")

        usage = Colors.strip_ansi(self.parser.format_usage()).strip()
        lines.append(f"\n```text\n{usage}\n```")

        global_options = [
            action
            for action in self.parser._actions
            if not isinstance(action, argparse._SubParsersAction) and action.option_strings
        ]
        if global_options:
            lines.append("\n**Global Options**\n")
            lines.append("| Flag | Type / Choices | Default | Description |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for action in global_options:
                flags = ", ".join(f"`{opt}`" for opt in action.option_strings)
                lines.append(
                    f"| {flags} | {_format_action_type(action)} | {_format_action_default(action)} | {_escape_md_cell(action.help)} |"
                )

        if self.commands:
            lines.append("\n**Commands**\n")
            lines.append("| Command | Aliases | Summary |")
            lines.append("| :--- | :--- | :--- |")
            for cmd in self.commands:
                aliases_cell = ", ".join(f"`{a}`" for a in cmd.aliases) if cmd.aliases else "—"
                summary = _escape_md_cell(cmd.help_text or cmd.description)
                lines.append(f"| `{cmd.name}` | {aliases_cell} | {summary} |")

            for cmd in self.commands:
                lines.append(
                    "\n"
                    + cmd.to_markdown(
                        heading_level=heading_level + 1,
                        prog_prefix=prog,
                        include_global_options=include_global_options,
                    )
                )

        if self.config.epilog:
            lines.append(f"\n---\n\n{Colors.strip_ansi(self.config.epilog).strip()}")

        return "\n".join(lines)

    def to_tools(self) -> list[ToolSpec]:
        r"""Flatten all executable commands registered on this CLI into [`ToolSpec`][clio.agent.ToolSpec] definitions.

        Returns:
            list[ToolSpec]: List of flattened LLM / MCP tool definitions.
        """
        tools: list[ToolSpec] = []
        for cmd in self.commands:
            tools.extend(cmd.to_tools())
        return tools

    def to_json_schema(self) -> dict[str, Any]:
        r"""Generate a complete JSON Schema and tool manifest for the CLI application.

        Returns:
            dict[str, Any]: Dictionary containing `$schema`, `prog`, `description`, `version`,
                hierarchical `commands`, and flattened `tools`.
        """
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "prog": self.parser.prog,
            "description": Colors.strip_ansi(self.config.description or "").strip(),
            "version": self.config.version,
            "commands": {cmd.name: cmd.to_json_schema() for cmd in self.commands},
            "tools": [tool.to_dict() for tool in self.to_tools()],
        }

    def to_skill_md(self) -> str:
        r"""Generate a zero-dependency Agent `SKILL.md` file (with YAML frontmatter) for the entire CLI.

        Returns:
            str: Formatted `SKILL.md` documentation string.
        """
        prog = self.parser.prog or "cli"
        skill_name = prog.replace(" ", "-").replace("_", "-").lower()
        raw_desc = Colors.strip_ansi(self.config.description or f"Execute {prog} CLI commands.").strip()
        summary_line = " ".join(raw_desc.splitlines())

        lines: list[str] = [
            "---",
            f"name: {skill_name}",
            f"description: {summary_line}",
            "---",
            "",
            f"# `{prog}` Agent Skill",
            "",
            "## Agent Execution & Introspection",
            "",
            "- Set `CLIO_AGENT=1` in the environment for concise, token-efficient error messages and to disable interactive `\r` progress spinners.",
        ]
        if self.config.enable_agent_flags:
            lines.extend(
                [
                    f"- Pass `--json` (e.g. `{prog} <command> --json`) to receive a structured JSON envelope (`ok`, `command`, `result`, `messages`, `error`).",
                    f"- Pass `--dry-run` (e.g. `{prog} <command> ... --dry-run`) to validate arguments and inspect resolved parameter bindings without side effects.",
                    f"- Run `{prog} --schema` (or `{prog} <command> --schema`) to inspect the full JSON Schema and tool definitions.",
                    f"- Run `{prog} --help-md` to render Markdown reference documentation.",
                    f"- Run `{prog} --mcp` to start a stdio Model Context Protocol (MCP) JSON-RPC 2.0 server.",
                ]
            )
        lines.append("")
        lines.append(self.to_markdown(heading_level=2))
        return "\n".join(lines)

    def serve_mcp(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        r"""Launch a blocking stdio Model Context Protocol (MCP) JSON-RPC 2.0 server for this CLI.

        Args:
            stdin (TextIO | None): Input stream for JSON-RPC messages. Defaults to `sys.stdin`.
            stdout (TextIO | None): Output stream for JSON-RPC responses. Defaults to `sys.stdout`.
        """
        McpServer(self).serve(stdin=stdin, stdout=stdout)

    def run(self, args: list[str] | None = None) -> Any:
        r"""Parse arguments and execute bound subcommand handler.

        Args:
            args (list[str] | None): Command line arguments to parse. Defaults to `sys.argv[1:]`.

        Returns:
            Any: Return value of the executed command, or [`AgentResult`][clio.agent.AgentResult]
                when `--json` or `--dry-run` is active.

        Raises:
            SystemExit: If a tracked exception in `catch_exceptions` occurs during execution.
        """
        raw_args = list(sys.argv[1:] if args is None else args)
        json_requested = self.config.enable_agent_flags and ("--json" in raw_args)

        prev_json = HelpOnErrorParser._ACTIVE_JSON_MODE
        prev_agent = HelpOnErrorParser._ACTIVE_AGENT_MODE
        HelpOnErrorParser._ACTIVE_JSON_MODE = json_requested
        HelpOnErrorParser._ACTIVE_AGENT_MODE = self.agent_mode
        HelpOnErrorParser._LAST_SUBPARSER = None
        try:
            parsed_args = self.parser.parse_args(args)
        finally:
            HelpOnErrorParser._ACTIVE_JSON_MODE = prev_json
            HelpOnErrorParser._ACTIVE_AGENT_MODE = prev_agent
            HelpOnErrorParser._LAST_SUBPARSER = None

        self.verbose = bool(getattr(parsed_args, "verbose", False))
        parsed_args.verbose = self.verbose
        is_json = bool(getattr(parsed_args, "_clio_json", False)) or json_requested
        is_dry_run = bool(getattr(parsed_args, "_clio_dry_run", False))
        matched_cmd: BaseCommand | None = getattr(parsed_args, "_clio_command", None)
        cmd_name = matched_cmd.name if matched_cmd is not None else getattr(parsed_args, "command", None)

        if is_dry_run:
            resolved = matched_cmd.resolve_arguments(parsed_args) if matched_cmd is not None else {}
            dry_result = AgentResult(
                ok=True,
                command=cmd_name,
                dry_run=True,
                arguments=resolved,
            )
            print(dry_result.to_json())
            return dry_result

        if is_json:
            if hasattr(parsed_args, "func"):
                caught_exc: Exception | None = None
                ret: Any = None
                with _OutputCaptureContext(self) as cap:
                    try:
                        ret = parsed_args.func(parsed_args)
                    except self.catch_exceptions as exc:
                        caught_exc = exc

                if caught_exc is not None:
                    exit_code = getattr(caught_exc, "exit_code", 1)
                    err_env = AgentResult(
                        ok=False,
                        command=cmd_name,
                        messages=tuple(cap.messages),
                        error=AgentErrorSpec(message=str(caught_exc), exit_code=exit_code),
                    )
                    print(err_env.to_json())
                    sys.exit(exit_code)

                ok_env = AgentResult(
                    ok=True,
                    command=cmd_name,
                    result=ret,
                    messages=tuple(cap.messages),
                )
                print(ok_env.to_json())
                return ok_env

            self.parser.print_help()
            return None

        if hasattr(parsed_args, "func"):
            try:
                return parsed_args.func(parsed_args)
            except self.catch_exceptions as exc:
                exit_code = getattr(exc, "exit_code", 1)
                self.exit(str(exc), code=exit_code)
        else:
            self.parser.print_help()
            return None

    def __enter__(self) -> Self:
        r"""Enter context manager scope returning CLI instance.

        Returns:
            Self: The active CLI host instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        r"""Exit context manager scope and clean up allocated resources.

        Args:
            exc_type (type[BaseException] | None): Exception type raised within context, if any.
            exc_val (BaseException | None): Exception instance raised within context, if any.
            exc_tb (types.TracebackType | None): Traceback object associated with exception, if any.

        Raises:
            SystemExit: Terminates process with standard exit codes for handled signals.
        """
        self.cleanup()
        if exc_type is KeyboardInterrupt:
            print("\n🛑 Cancelled by user.", file=sys.stderr)
            sys.exit(1)
        elif exc_type is BrokenPipeError:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(130)
        elif exc_type is PermissionError:
            print(f"🔒 Permission denied: {exc_val}", file=sys.stderr)
            sys.exit(1)
        elif exc_type is FileNotFoundError:
            print(f"📄 File not found: {exc_val}", file=sys.stderr)
            sys.exit(1)

    def exit(self, msg: str, code: int = 1) -> None:
        r"""Print error message to stderr and terminate process.

        Args:
            msg (str): Error message string to report.
            code (int): Process exit code to pass to `sys.exit`. Defaults to 1.

        Raises:
            SystemExit: Exits process with specified status code.
        """
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(code)

    def __del__(self) -> None:
        r"""Clean up resource handles upon garbage collection."""
        self.cleanup()

    def cleanup(self) -> None:
        r"""Close any open file handles tracked by this CLI instance."""
        if not self._open_handles:
            return
        protected = (sys.stdout, sys.stdin, sys.stderr, sys.stdout.buffer, sys.stderr.buffer)
        for handle in self._open_handles:
            if handle not in protected:
                handle.close()
        self._open_handles.clear()

    def msg(self, msg: str | None, **kwargs: Any) -> None:
        r"""Print informational message to stderr if verbose mode is enabled.

        Args:
            msg (str | None): Message string to print.
            **kwargs (Any): Additional keyword arguments passed to `print`.
        """
        if self.verbose:
            if self._active_message_sink is not None and msg is not None:
                self._active_message_sink.append(AgentMessage(level="info", text=Colors.strip_ansi(str(msg)).strip()))
                return
            print(msg, file=sys.stderr, **kwargs)

    def progress(self, iterable: Iterable[Any], msg: str) -> Iterable[Any]:
        r"""Wrap iterable to display live progress counter on stderr when verbose mode is active.

        Uses a fast `yield from` pass-through when `verbose` is disabled, and suppresses
        carriage-return `\r` animations when `agent_mode` (`CLIO_AGENT=1`) or `--json` is active.

        Args:
            iterable (Iterable[Any]): Source collection or generator to iterate over.
            msg (str): Prefix status message displayed during iteration.

        Returns:
            Iterable[Any]: Generator yielding items from original iterable while rendering progress.
        """
        if not self.verbose:
            yield from iterable
            return

        try:
            total: int | str = len(iterable)  # type: ignore
        except TypeError:
            total = "?"

        if self._active_message_sink is not None:
            self._active_message_sink.append(
                AgentMessage(level="info", text=f"{Colors.strip_ansi(msg).strip()} ({total})")
            )
            yield from iterable
            return

        if self.agent_mode:
            print(f"{msg} ({total})", file=sys.stderr)
            yield from iterable
            return

        stderr = sys.stderr
        for i, item in enumerate(iterable, start=1):
            print(f"\r{msg} {i}/{total}", end="", file=stderr, flush=True)
            yield item

        print(file=stderr)

    def open_file(self, file: str | os.PathLike[str], mode: str = "rb") -> IO[Any]:
        r"""Open file path or return standard stdout stream handle with tracking.

        Args:
            file (str | os.PathLike[str]): File path or `"-"` / `"stdout"` for standard output stream.
            mode (str): File open mode flag. Defaults to `"rb"`.

        Returns:
            IO[Any]: Open file handle or system stream object.
        """
        if str(file) in ("-", "stdout"):
            return sys.stdout.buffer if "b" in mode else sys.stdout

        handle = open(file, mode)
        self._open_handles.append(handle)
        return handle
