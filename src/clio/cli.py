import inspect
import json
import os
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
from clio.color import Colors
from clio.command import (
    BaseCommand,
    Command,
    CommandTarget,
    _escape_md_cell,
    _format_spec_default,
    _format_spec_description,
    _format_spec_type,
    _LazyCommandEntry,
)
from clio.docstr import DocstringParser
from clio.parser import _DEFAULT_FORMATTER, CliError, HelpFormatter, ParseError, ScanResult, _fuzzy_suggest
from clio.spinner import _DEFAULT_SPINNER_CONFIG, Spinner, SpinnerConfig


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
        formatter (HelpFormatter): Immutable ANSI help and usage formatter specification.
        package_meta (PackageMeta | None): Resolved `importlib.metadata` package specification, if any.
        enable_agent_flags (bool): Whether opt-in agent CLI flags (`--json`, `--dry-run`, `--schema`,
            `--help-md`, `--skill`, `--mcp`) are registered.
    """

    prog: str | None = None
    description: str | None = None
    epilog: str | None = None
    version: str | None = None
    catch_exceptions: tuple[type[Exception], ...] = (CliError,)
    formatter: HelpFormatter = _DEFAULT_FORMATTER
    package_meta: PackageMeta | None = None
    enable_agent_flags: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedExecution:
    r"""Internal resolved leaf command and scan state ready for sync or async execution."""

    command: Command
    cmd_label: str
    scan: ScanResult
    is_json: bool


def _has_unquoted_json_flag(tokens: Sequence[str], *, enabled: bool) -> bool:
    r"""Return `True` if `--json` appears before any `--` end-of-options sentinel."""
    if not enabled:
        return False
    for tok in tokens:
        if tok == "--":
            break
        if tok == "--json":
            return True
    return False


def _builtin_pool(*, has_version: bool, enable_agent_flags: bool, allow_mcp: bool) -> tuple[str, ...]:
    r"""Return built-in global/agent flags valid at a non-leaf routing level."""
    flags: list[str] = ["-h", "--help", "-V", "--verbose"]
    if has_version:
        flags.extend(("-v", "--version"))
    if enable_agent_flags:
        flags.extend(("--json", "--dry-run", "--schema", "--help-md", "--skill"))
        if allow_mcp:
            flags.append("--mcp")
    return tuple(flags)


class Cli:
    r"""Root Command Line Interface runner, lazy subcommand router, and context manager.

    Coordinates zero-`argparse` lazy subcommand routing, single-command execution,
    synchronous and `async` command invocation, structured agent introspection, safe file
    handle tracking, error trapping, and progress/spinner reporting.

    Attributes:
        config (CliConfig): Immutable configuration specification for this CLI instance.
        prog (str): Resolved program name displayed in usage and help output.
        verbose (bool): Flag indicating whether verbose logging is enabled.
        catch_exceptions (tuple[type[Exception], ...]): Exception types trapped gracefully.
    """

    def __init__(
        self,
        prog: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        version: str | None = None,
        catch_exceptions: Iterable[type[Exception]] = (),
        formatter: HelpFormatter = _DEFAULT_FORMATTER,
        commands: Iterable[CommandTarget] = (),
        package: str | None = None,
        *,
        command: CommandTarget | None = None,
        enable_agent_flags: bool = False,
        agent_mode: bool | None = None,
    ) -> None:
        r"""Initialize root CLI application instance.

        Automatically queries `importlib.metadata` when `package` is provided (or when `prog` or the
        calling package matches an installed distribution) to populate omitted `prog`, `description`,
        and `version` fields.

        Args:
            prog (str | None): Program name displayed in usage (defaults to package name, root command,
                or `sys.argv[0]`).
            description (str | None): Header description text displayed in top-level help.
            epilog (str | None): Footer epilog text displayed in top-level help.
            version (str | None): Program version string. If provided (or resolved via `importlib.metadata`),
                registers `-v`/`--version`.
            catch_exceptions (Iterable[type[Exception]]): Domain exception classes to trap and
                format cleanly to `sys.stderr` without a traceback. `CliError` is always included.
            formatter (HelpFormatter): Help formatter specification applied across root and subcommands.
            commands (Iterable[CommandTarget]): Optional sequence of commands (classes, dataclasses,
                or instances) to register lazily in multi-command tree mode.
            package (str | None): Optional distribution package name to query via `importlib.metadata`.
            command (CommandTarget | None): Optional single root command (mutually exclusive with `commands`).
            enable_agent_flags (bool): Opt in to registering built-in agent CLI flags (`--json`,
                `--dry-run`, `--schema`, `--help-md`, `--skill`, `--mcp`).
            agent_mode (bool | None): Override `CLIO_AGENT=1` environment detection for concise
                error output and disabled `\r` spinners.

        Raises:
            ValueError: If both `command` and `commands` are supplied simultaneously.
        """
        self._open_handles: list[IO[Any]] = []
        cmd_list = list(commands)
        if command is not None and cmd_list:
            raise ValueError("Cannot specify both 'command' (single-command mode) and 'commands' (subcommand mode).")

        candidate_pkg = package
        if candidate_pkg is None and prog is None:
            if type(self) is not Cli:
                mod_root = type(self).__module__.split(".", 1)[0]
                if mod_root not in ("__main__", "builtins", "clio"):
                    candidate_pkg = mod_root
            if candidate_pkg is None and (frame := inspect.currentframe()) is not None:
                caller = frame.f_back
                while caller is not None:
                    caller_pkg = caller.f_globals.get("__package__")
                    if isinstance(caller_pkg, str) and caller_pkg:
                        pkg_root = caller_pkg.split(".", 1)[0]
                        if pkg_root == "clio":
                            caller = caller.f_back
                            continue
                        if pkg_root not in ("__main__", "builtins"):
                            candidate_pkg = pkg_root
                    break

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

        self._root_entry: _LazyCommandEntry | None = _LazyCommandEntry(command) if command is not None else None
        self._entries: list[_LazyCommandEntry] = []
        self._route_map: dict[str, _LazyCommandEntry] = {}

        self.config = CliConfig(
            prog=prog,
            description=description,
            epilog=epilog,
            version=version,
            catch_exceptions=(CliError, *catch_exceptions),
            formatter=formatter,
            package_meta=pkg_meta,
            enable_agent_flags=enable_agent_flags,
        )
        self.prog: str = (
            self.config.prog
            or (self._root_entry.name if self._root_entry is not None else "")
            or os.path.basename(sys.argv[0])
            or "cli"
        )
        self.verbose = False
        self._agent_mode_override = agent_mode
        self._active_message_sink: list[AgentMessage] | None = None
        self.catch_exceptions: tuple[type[Exception], ...] = self.config.catch_exceptions

        if cmd_list:
            self.add_commands(*cmd_list)

    @property
    def agent_mode(self) -> bool:
        r"""Whether token-efficient agent runtime mode (`CLIO_AGENT=1`) is active."""
        if self._agent_mode_override is not None:
            return self._agent_mode_override
        return _detect_agent_mode()

    @property
    def root_command(self) -> BaseCommand | None:
        r"""Materialize and return the single root command when configured via `Cli(command=...)`, else `None`."""
        if self._root_entry is None:
            return None
        return self._root_entry.resolve(self)

    @property
    def commands(self) -> tuple[BaseCommand, ...]:
        r"""Materialize and return all registered top-level subcommands."""
        return tuple(entry.resolve(self) for entry in self._entries)

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
        formatter: HelpFormatter = _DEFAULT_FORMATTER,
        commands: Iterable[CommandTarget] = (),
        command: CommandTarget | None = None,
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
            formatter (HelpFormatter): Help formatter specification.
            commands (Iterable[CommandTarget]): Subcommands to register lazily.
            command (CommandTarget | None): Optional single root command.
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
            formatter=formatter,
            commands=commands,
            package=package,
            command=command,
            enable_agent_flags=enable_agent_flags,
            agent_mode=agent_mode,
        )

    def add_command(self, command: CommandTarget) -> Self:
        r"""Lazily index and attach a top-level command (instance, class, or `@dataclass`) without compiling specs.

        Args:
            command (CommandTarget): Subcommand instance, `BaseCommand` subclass, or `@dataclass`.

        Returns:
            Self: The CLI instance for method chaining.

        Raises:
            ValueError: If called on a single-command `Cli(command=...)` instance.
        """
        if self._root_entry is not None:
            raise ValueError("Cannot add subcommands to a single-command Cli(command=...) instance.")
        entry = _LazyCommandEntry(command)
        self._entries.append(entry)
        self._route_map[entry.name] = entry
        for alias in entry.aliases:
            self._route_map[alias] = entry
        return self

    def add_commands(self, *commands: CommandTarget) -> Self:
        r"""Lazily index and attach multiple top-level commands (instances, classes, or `@dataclass` types).

        Args:
            *commands (CommandTarget): Subcommand classes, dataclasses, or instances to register.

        Returns:
            Self: The CLI instance for method chaining.
        """
        for command in commands:
            self.add_command(command)
        return self

    def spinner(
        self,
        message: str,
        *,
        config: SpinnerConfig = _DEFAULT_SPINNER_CONFIG,
        stream: TextIO | None = None,
        show_elapsed: bool = True,
    ) -> Spinner:
        r"""Create an agent-aware [`Spinner`][clio.spinner.Spinner] inheriting this CLI's `agent_mode` setting.

        Args:
            message (str): Initial status message displayed alongside the spinner.
            config (SpinnerConfig): Visual frames, interval, and styling configuration.
            stream (TextIO | None): Target output stream (defaults to `sys.stderr`).
            show_elapsed (bool): Whether to append elapsed seconds on completion.

        Returns:
            Spinner: Context manager supporting both `with` and `async with`.
        """
        return Spinner(
            message,
            config=config,
            stream=stream,
            agent_mode=self._agent_mode_override,
            show_elapsed=show_elapsed,
        )

    def format_usage(self) -> str:
        r"""Render the single-line ANSI usage synopsis for the root CLI."""
        if self._root_entry is not None:
            root_cmd = self.root_command
            assert root_cmd is not None
            scan_plan = root_cmd.get_scan_plan()
            return self.config.formatter.format_usage(
                self.prog,
                positionals=scan_plan.positionals,
                has_options=True,
                has_subcommands=root_cmd.has_subcommands(),
            )
        return self.config.formatter.format_usage(
            self.prog,
            positionals=(),
            has_options=True,
            has_subcommands=bool(self._entries),
        )

    def format_help(self) -> str:
        r"""Render complete ANSI help output for the root CLI without evaluating subcommand annotations."""
        if self._root_entry is not None:
            root_cmd = self.root_command
            assert root_cmd is not None
            desc = self.config.description if self.config.description is not None else root_cmd.description
            return self.config.formatter.format_help(
                self.prog,
                description=desc,
                epilog=self.config.epilog,
                scan_plan=root_cmd.get_scan_plan(),
                subcommands=root_cmd._subcommand_help_rows(),
                subcommands_title="💬 Commands",
                has_version=self.config.version is not None,
                enable_agent_flags=self.config.enable_agent_flags,
                allow_mcp=self.config.enable_agent_flags,
            )

        sub_rows = tuple((entry.name, entry.aliases, entry.summary()) for entry in self._entries)
        return self.config.formatter.format_help(
            self.prog,
            description=self.config.description,
            epilog=self.config.epilog,
            scan_plan=None,
            subcommands=sub_rows,
            subcommands_title="💬 Commands",
            has_version=self.config.version is not None,
            enable_agent_flags=self.config.enable_agent_flags,
            allow_mcp=self.config.enable_agent_flags,
        )

    def _global_markdown_rows(self, *, shadowed: Mapping[str, Any] | None = None) -> list[str]:
        skip = shadowed or {}
        rows: list[str] = []
        if "-h" not in skip and "--help" not in skip:
            rows.append("| `-h`, `--help` | `bool` | `False` | Show this help message and exit |")
        if "-V" not in skip and "--verbose" not in skip:
            rows.append("| `-V`, `--verbose` | `bool` | `False` | Enable verbose output/progress |")
        if self.config.version is not None and "-v" not in skip and "--version" not in skip:
            rows.append("| `-v`, `--version` | `str` | `None` | Show program's version number and exit |")
        if self.config.enable_agent_flags:
            agent_specs = (
                ("--json", "Emit structured JSON execution envelope"),
                ("--dry-run", "Validate arguments and inspect resolved CallPlan without executing"),
                ("--schema", "Print JSON Schema specification and exit"),
                ("--help-md", "Print Markdown reference documentation and exit"),
                ("--skill", "Print Agent SKILL.md specification and exit"),
                ("--mcp", "Start stdio Model Context Protocol (MCP) JSON-RPC server"),
            )
            for flag, desc in agent_specs:
                if flag not in skip:
                    rows.append(f"| `{flag}` | `bool` | `False` | {desc} |")
        return rows

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
        prog = self.prog
        hashes = "#" * max(1, heading_level)
        lines: list[str] = [f"{hashes} `{prog}`"]

        root_cmd = self.root_command
        desc_text = self.config.description or (root_cmd.description if root_cmd is not None else None)
        if desc_text:
            lines.append(f"\n{Colors.strip_ansi(desc_text).strip()}")

        usage = Colors.strip_ansi(self.format_usage()).strip()
        lines.append(f"\n```text\n{usage}\n```")

        if root_cmd is not None:
            scan_plan = root_cmd.get_scan_plan()
            if scan_plan.positionals:
                lines.append("\n**Positional Arguments**\n")
                lines.append("| Argument | Type / Choices | Description |")
                lines.append("| :--- | :--- | :--- |")
                for pos in scan_plan.positionals:
                    lines.append(f"| `{pos.name}` | {_format_spec_type(pos)} | {_format_spec_description(pos)} |")

            option_rows: list[str] = []
            for opt in scan_plan.options:
                flag_list = list(opt.option_flags)
                if opt.show_negated_in_help and opt.negated_flag is not None and opt.negated_flag not in flag_list:
                    flag_list.append(opt.negated_flag)
                flags = ", ".join(f"`{f}`" for f in flag_list)
                option_rows.append(
                    f"| {flags} | {_format_spec_type(opt)} | {_format_spec_default(opt)} | {_format_spec_description(opt)} |"
                )
            option_rows.extend(self._global_markdown_rows(shadowed=scan_plan.flag_map))
            if option_rows:
                lines.append("\n**Options**\n")
                lines.append("| Flag | Type / Choices | Default | Description |")
                lines.append("| :--- | :--- | :--- | :--- |")
                lines.extend(option_rows)
        else:
            global_rows = self._global_markdown_rows()
            if global_rows:
                lines.append("\n**Global Options**\n")
                lines.append("| Flag | Type / Choices | Default | Description |")
                lines.append("| :--- | :--- | :--- | :--- |")
                lines.extend(global_rows)

            resolved_cmds = self.commands
            if resolved_cmds:
                lines.append("\n**Commands**\n")
                lines.append("| Command | Aliases | Summary |")
                lines.append("| :--- | :--- | :--- |")
                for cmd in resolved_cmds:
                    aliases_cell = ", ".join(f"`{a}`" for a in cmd.aliases) if cmd.aliases else "—"
                    summary = _escape_md_cell(cmd.help_text or cmd.description)
                    lines.append(f"| `{cmd.name}` | {aliases_cell} | {summary} |")

                for cmd in resolved_cmds:
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
        if self.root_command is not None:
            return self.root_command.to_tools()
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
        if self.root_command is not None:
            root_cmd = self.root_command
            desc = Colors.strip_ansi(self.config.description or root_cmd.description or "").strip()
            commands_map = {root_cmd.name: root_cmd.to_json_schema()}
        else:
            desc = Colors.strip_ansi(self.config.description or "").strip()
            commands_map = {cmd.name: cmd.to_json_schema() for cmd in self.commands}

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "prog": self.prog,
            "description": desc,
            "version": self.config.version,
            "commands": commands_map,
            "tools": [tool.to_dict() for tool in self.to_tools()],
        }

    def to_skill_md(self) -> str:
        r"""Generate a zero-dependency Agent `SKILL.md` file (with YAML frontmatter) for the entire CLI.

        Returns:
            str: Formatted `SKILL.md` documentation string.
        """
        prog = self.prog or "cli"
        skill_name = prog.replace(" ", "-").replace("_", "-").lower()
        root_cmd = self.root_command
        raw_desc = Colors.strip_ansi(
            self.config.description
            or (root_cmd.description if root_cmd is not None else "")
            or f"Execute {prog} CLI commands."
        ).strip()
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
            cmd_placeholder = "" if root_cmd is not None else " <command>"
            lines.extend(
                [
                    f"- Pass `--json` (e.g. `{prog}{cmd_placeholder} --json`) to receive a structured JSON envelope (`ok`, `command`, `result`, `messages`, `error`).",
                    f"- Pass `--dry-run` (e.g. `{prog}{cmd_placeholder} ... --dry-run`) to validate arguments and inspect resolved parameter bindings without side effects.",
                    f"- Run `{prog} --schema` to inspect the full JSON Schema and tool definitions.",
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

    def _emit_eager_action(
        self,
        action: str,
        *,
        target_cmd: BaseCommand | None,
        prog_prefix: str,
        is_single_root: bool,
    ) -> None:
        r"""Execute an eager zero-argument introspection flag and exit with status 0."""
        use_root = target_cmd is None or is_single_root
        match action:
            case "help":
                help_out = self.format_help() if use_root else target_cmd.format_help(prog_prefix=prog_prefix)  # type: ignore[union-attr]
                print(help_out, end="")
            case "version":
                print(f"{self.prog} {self.config.version}")
            case "schema":
                payload = self.to_json_schema() if use_root else target_cmd.to_json_schema()  # type: ignore[union-attr]
                print(json.dumps(payload, indent=2))
            case "help_md":
                md = self.to_markdown() if use_root else target_cmd.to_markdown(prog_prefix=prog_prefix)  # type: ignore[union-attr]
                print(md)
            case "skill":
                skill = (
                    self.to_skill_md()
                    if use_root
                    else target_cmd.to_skill_md(  # type: ignore[union-attr]
                        prog_prefix=prog_prefix,
                        enable_agent_flags=self.config.enable_agent_flags,
                    )
                )
                print(skill)
            case "mcp":
                self.serve_mcp()
        sys.exit(0)

    def _handle_parse_error(
        self,
        err: ParseError,
        *,
        json_requested: bool,
        target_cmd: BaseCommand | None,
        prog_prefix: str,
        is_single_root: bool,
    ) -> None:
        r"""Format a `ParseError` according to `--json`, `CLIO_AGENT=1`, or interactive TTY mode and exit(2)."""
        if json_requested:
            err_result = AgentResult(
                ok=False,
                command=err.command_path or self.prog,
                error=AgentErrorSpec(
                    message=Colors.strip_ansi(str(err)).strip(),
                    exit_code=err.exit_code,
                    usage=err.usage or None,
                    suggestions=err.suggestions,
                ),
            )
            print(err_result.to_json())
            sys.exit(err.exit_code)

        display_msg = str(err)
        if err.suggestions:
            display_msg += f"\n    💡 Did you mean '{Colors.info(err.suggestions[0])}'?"

        if self.agent_mode:
            cmd_path_str = err.command_path or self.prog
            usage_line = err.usage or Colors.strip_ansi(self.format_usage()).strip()
            hint_line = f"    Run '{cmd_path_str} --help' for full help."
            print(f"{usage_line}\n{Colors.error('❌ Error:')} {display_msg}\n{hint_line}", file=sys.stderr)
            sys.exit(err.exit_code)

        help_text = (
            self.format_help()
            if (target_cmd is None or is_single_root)
            else target_cmd.format_help(prog_prefix=prog_prefix)
        )
        print(help_text, end="", file=sys.stderr)
        print(f"\n{Colors.error('❌ Error:')} {display_msg}", file=sys.stderr)
        sys.exit(err.exit_code)

    def _prepare_execution(self, args: Sequence[str] | None) -> _PreparedExecution | AgentResult | None:
        r"""Route subcommands lazily, scan tokens, and resolve dry-run or eager actions."""
        raw_tokens = list(sys.argv[1:] if args is None else args)
        enable_agent = self.config.enable_agent_flags
        has_version = self.config.version is not None
        json_requested = _has_unquoted_json_flag(raw_tokens, enabled=enable_agent)

        verbose = False
        json_mode = False
        dry_run = False
        target_cmd: BaseCommand | None = None
        prog_prefix = ""
        is_single_root = self._root_entry is not None

        try:
            if is_single_root:
                current_cmd = self.root_command
                assert current_cmd is not None
                target_cmd = current_cmd
                prog_path = self.prog
                cmd_path: tuple[str, ...] = (current_cmd.name or self.prog,)
                tokens: Sequence[str] = raw_tokens
            else:
                root_usage = Colors.strip_ansi(self.format_usage()).strip()
                idx = 0
                total_root = len(raw_tokens)
                after_dash = False
                matched_entry: _LazyCommandEntry | None = None

                while idx < total_root:
                    tok = raw_tokens[idx]
                    if not after_dash and tok == "--":
                        after_dash = True
                        idx += 1
                        continue
                    if not after_dash:
                        if tok in ("-h", "--help"):
                            self._emit_eager_action("help", target_cmd=None, prog_prefix="", is_single_root=False)
                        if has_version and tok in ("-v", "--version"):
                            self._emit_eager_action("version", target_cmd=None, prog_prefix="", is_single_root=False)
                        if tok in ("-V", "--verbose"):
                            verbose = True
                            idx += 1
                            continue
                        if enable_agent:
                            match tok:
                                case "--json":
                                    json_mode = True
                                    idx += 1
                                    continue
                                case "--dry-run":
                                    dry_run = True
                                    idx += 1
                                    continue
                                case "--schema":
                                    self._emit_eager_action(
                                        "schema", target_cmd=None, prog_prefix="", is_single_root=False
                                    )
                                case "--help-md":
                                    self._emit_eager_action(
                                        "help_md", target_cmd=None, prog_prefix="", is_single_root=False
                                    )
                                case "--skill":
                                    self._emit_eager_action(
                                        "skill", target_cmd=None, prog_prefix="", is_single_root=False
                                    )
                                case "--mcp":
                                    self._emit_eager_action(
                                        "mcp", target_cmd=None, prog_prefix="", is_single_root=False
                                    )

                        if len(tok) > 1 and tok[0] == "-":
                            flag_part = tok.split("=", 1)[0]
                            pool = _builtin_pool(
                                has_version=has_version,
                                enable_agent_flags=enable_agent,
                                allow_mcp=enable_agent,
                            )
                            raise ParseError(
                                f"unrecognized arguments: {tok}",
                                usage=root_usage,
                                suggestions=_fuzzy_suggest(flag_part, pool),
                                command_path=self.prog,
                            )

                    matched_entry = self._route_map.get(tok)
                    if matched_entry is None:
                        all_names = [name for e in self._entries for name in (e.name, *e.aliases)]
                        choices_str = ", ".join(repr(e.name) for e in self._entries)
                        msg = (
                            f"argument command: invalid choice: {tok!r} (choose from {choices_str})"
                            if choices_str
                            else f"unrecognized arguments: {tok}"
                        )
                        raise ParseError(
                            msg,
                            usage=root_usage,
                            suggestions=_fuzzy_suggest(tok, all_names),
                            command_path=self.prog,
                        )
                    idx += 1
                    break

                if matched_entry is None:
                    if not self._entries:
                        print(self.format_help(), end="")
                        return None
                    raise ParseError(
                        "the following arguments are required: command",
                        usage=root_usage,
                        command_path=self.prog,
                    )

                current_cmd = matched_entry.resolve(self)
                target_cmd = current_cmd
                prog_prefix = self.prog
                prog_path = f"{self.prog} {current_cmd.name}"
                cmd_path = (current_cmd.name,)
                tokens = raw_tokens[idx:]

            # Descend through nested CommandGroups or hybrid Commands with subcommands
            while current_cmd.has_subcommands():
                sub_idx = 0
                sub_total = len(tokens)
                descended = False
                cur_usage = Colors.strip_ansi(
                    self.format_usage() if is_single_root else current_cmd.format_usage(prog_prefix=prog_prefix)
                ).strip()

                while sub_idx < sub_total:
                    tok = tokens[sub_idx]
                    if tok == "--":
                        break
                    if tok in ("-h", "--help"):
                        self._emit_eager_action(
                            "help",
                            target_cmd=current_cmd,
                            prog_prefix=prog_prefix,
                            is_single_root=is_single_root,
                        )
                    if is_single_root and has_version and tok in ("-v", "--version"):
                        self._emit_eager_action(
                            "version",
                            target_cmd=current_cmd,
                            prog_prefix=prog_prefix,
                            is_single_root=is_single_root,
                        )
                    if tok in ("-V", "--verbose"):
                        verbose = True
                        sub_idx += 1
                        continue
                    if enable_agent:
                        match tok:
                            case "--json":
                                json_mode = True
                                sub_idx += 1
                                continue
                            case "--dry-run":
                                dry_run = True
                                sub_idx += 1
                                continue
                            case "--schema":
                                self._emit_eager_action(
                                    "schema",
                                    target_cmd=current_cmd,
                                    prog_prefix=prog_prefix,
                                    is_single_root=is_single_root,
                                )
                            case "--help-md":
                                self._emit_eager_action(
                                    "help_md",
                                    target_cmd=current_cmd,
                                    prog_prefix=prog_prefix,
                                    is_single_root=is_single_root,
                                )
                            case "--skill":
                                self._emit_eager_action(
                                    "skill",
                                    target_cmd=current_cmd,
                                    prog_prefix=prog_prefix,
                                    is_single_root=is_single_root,
                                )
                            case "--mcp" if is_single_root:
                                self._emit_eager_action(
                                    "mcp",
                                    target_cmd=current_cmd,
                                    prog_prefix=prog_prefix,
                                    is_single_root=is_single_root,
                                )

                    if len(tok) > 1 and tok[0] == "-":
                        if current_cmd._is_executable():
                            break
                        flag_part = tok.split("=", 1)[0]
                        pool = _builtin_pool(
                            has_version=is_single_root and has_version,
                            enable_agent_flags=enable_agent,
                            allow_mcp=is_single_root and enable_agent,
                        )
                        raise ParseError(
                            f"unrecognized arguments: {tok}",
                            usage=cur_usage,
                            suggestions=_fuzzy_suggest(flag_part, pool),
                            command_path=prog_path,
                        )

                    child = current_cmd._match_subcommand(tok)
                    if child is not None:
                        prog_prefix = prog_path
                        prog_path = f"{prog_path} {child.name}"
                        cmd_path = (*cmd_path, child.name)
                        current_cmd = child
                        target_cmd = child
                        tokens = tokens[sub_idx + 1 :]
                        is_single_root = False
                        descended = True
                        break

                    if current_cmd._is_executable():
                        break

                    all_sub_names = [name for e in current_cmd._sub_entries for name in (e.name, *e.aliases)]
                    choices_str = ", ".join(repr(n) for n in current_cmd._subcommand_names())
                    raise ParseError(
                        f"argument {current_cmd.name}_subcommand: invalid choice: {tok!r} (choose from {choices_str})",
                        usage=cur_usage,
                        suggestions=_fuzzy_suggest(tok, all_sub_names),
                        command_path=prog_path,
                    )

                if descended:
                    continue

                if not current_cmd._is_executable():
                    raise ParseError(
                        f"the following arguments are required: {current_cmd.name}_subcommand",
                        usage=cur_usage,
                        command_path=prog_path,
                    )

                tokens = tokens[sub_idx:]
                break

            # Leaf ScanPlan execution
            scan_plan = current_cmd.get_scan_plan()
            leaf_usage = Colors.strip_ansi(
                self.format_usage() if is_single_root else current_cmd.format_usage(prog_prefix=prog_prefix)
            ).strip()

            # If leaf command defines its own `--json` user flag, do not treat `--json` as agent envelope mode
            if "--json" in scan_plan.flag_map and not json_mode:
                json_requested = False

            scan = scan_plan.parse(
                tokens,
                prog=prog_path,
                usage=leaf_usage,
                allow_version=is_single_root and has_version,
                enable_agent_flags=enable_agent,
                allow_mcp=is_single_root and enable_agent,
            )

        except ParseError as err:
            self._handle_parse_error(
                err,
                json_requested=json_requested,
                target_cmd=target_cmd,
                prog_prefix=prog_prefix,
                is_single_root=is_single_root,
            )
            raise AssertionError("unreachable") from err  # pragma: no cover

        if scan.eager_action is not None:
            self._emit_eager_action(
                scan.eager_action,
                target_cmd=current_cmd,
                prog_prefix=prog_prefix,
                is_single_root=is_single_root,
            )

        self.verbose = verbose or scan.verbose
        is_json = json_mode or scan.json_mode
        is_dry_run = dry_run or scan.dry_run
        cmd_label = " ".join(cmd_path)

        if is_dry_run:
            resolved = current_cmd.resolve_arguments(scan.values)
            dry_result = AgentResult(
                ok=True,
                command=cmd_label,
                dry_run=True,
                arguments=resolved,
            )
            print(dry_result.to_json())
            return dry_result

        assert isinstance(current_cmd, Command)
        return _PreparedExecution(
            command=current_cmd,
            cmd_label=cmd_label,
            scan=scan,
            is_json=is_json,
        )

    def run(self, args: Sequence[str] | None = None) -> Any:
        r"""Parse CLI tokens and execute the matched command handler.

        Synchronous commands execute directly with zero `asyncio` import overhead.
        Coroutine (`async def`) commands are automatically executed via a lazy `asyncio.run()` call.

        Args:
            args (Sequence[str] | None): Command line arguments to parse. Defaults to `sys.argv[1:]`.

        Returns:
            Any: Return value of the executed command, or [`AgentResult`][clio.agent.AgentResult]
                when `--json` or `--dry-run` is active.

        Raises:
            SystemExit: If a parsing error or tracked exception in `catch_exceptions` occurs.
        """
        prepared = self._prepare_execution(args)
        if not isinstance(prepared, _PreparedExecution):
            return prepared

        if prepared.is_json:
            caught_exc: Exception | None = None
            ret: Any = None
            with _OutputCaptureContext(self) as cap:
                try:
                    ret = prepared.command._invoke_values(prepared.scan.values)
                    if inspect.iscoroutine(ret):
                        import asyncio

                        ret = asyncio.run(ret)
                except self.catch_exceptions as exc:
                    caught_exc = exc

            if caught_exc is not None:
                exit_code = getattr(caught_exc, "exit_code", 1)
                err_env = AgentResult(
                    ok=False,
                    command=prepared.cmd_label,
                    messages=tuple(cap.messages),
                    error=AgentErrorSpec(message=str(caught_exc), exit_code=exit_code),
                )
                print(err_env.to_json())
                sys.exit(exit_code)

            ok_env = AgentResult(
                ok=True,
                command=prepared.cmd_label,
                result=ret,
                messages=tuple(cap.messages),
            )
            print(ok_env.to_json())
            return ok_env

        try:
            ret = prepared.command._invoke_values(prepared.scan.values)
            if inspect.iscoroutine(ret):
                import asyncio

                return asyncio.run(ret)
            return ret
        except self.catch_exceptions as exc:
            exit_code = getattr(exc, "exit_code", 1)
            self.exit(str(exc), code=exit_code)

    async def run_async(self, args: Sequence[str] | None = None) -> Any:
        r"""Parse CLI tokens and asynchronously execute the matched command on the active event loop.

        Supports both synchronous and `async def` commands seamlessly without starting a nested event loop.

        Args:
            args (Sequence[str] | None): Command line arguments to parse. Defaults to `sys.argv[1:]`.

        Returns:
            Any: Return value of the executed command, or [`AgentResult`][clio.agent.AgentResult]
                when `--json` or `--dry-run` is active.
        """
        prepared = self._prepare_execution(args)
        if not isinstance(prepared, _PreparedExecution):
            return prepared

        if prepared.is_json:
            caught_exc: Exception | None = None
            ret: Any = None
            with _OutputCaptureContext(self) as cap:
                try:
                    ret = prepared.command._invoke_values(prepared.scan.values)
                    if inspect.iscoroutine(ret):
                        ret = await ret
                except self.catch_exceptions as exc:
                    caught_exc = exc

            if caught_exc is not None:
                exit_code = getattr(caught_exc, "exit_code", 1)
                err_env = AgentResult(
                    ok=False,
                    command=prepared.cmd_label,
                    messages=tuple(cap.messages),
                    error=AgentErrorSpec(message=str(caught_exc), exit_code=exit_code),
                )
                print(err_env.to_json())
                sys.exit(exit_code)

            ok_env = AgentResult(
                ok=True,
                command=prepared.cmd_label,
                result=ret,
                messages=tuple(cap.messages),
            )
            print(ok_env.to_json())
            return ok_env

        try:
            ret = prepared.command._invoke_values(prepared.scan.values)
            if inspect.iscoroutine(ret):
                return await ret
            return ret
        except self.catch_exceptions as exc:
            exit_code = getattr(exc, "exit_code", 1)
            self.exit(str(exc), code=exit_code)

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
        if exc_type is None:
            return
        match exc_type:
            case _ if issubclass(exc_type, KeyboardInterrupt):
                print("\n🛑 Cancelled by user.", file=sys.stderr)
                sys.exit(1)
            case _ if issubclass(exc_type, BrokenPipeError):
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
                sys.exit(130)
            case _ if issubclass(exc_type, PermissionError):
                print(f"🔒 Permission denied: {exc_val}", file=sys.stderr)
                sys.exit(1)
            case _ if issubclass(exc_type, FileNotFoundError):
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
