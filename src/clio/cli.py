import argparse
import inspect
import os
import re
import sys
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import IO, Any, ClassVar, Self

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
    """

    prog: str | None = None
    description: str | None = None
    epilog: str | None = None
    version: str | None = None
    catch_exceptions: tuple[type[Exception], ...] = (CliError,)
    formatter_class: type[argparse.HelpFormatter] = ColorHelpFormatter
    package_meta: PackageMeta | None = None


class HelpOnErrorParser(argparse.ArgumentParser):
    r"""Argument parser that outputs full help text and suggestions on error.

    Extends `argparse.ArgumentParser` to print help menu to
    `sys.stderr` when invalid arguments or choices are encountered, suggesting close matches.
    """

    _INVALID_CHOICE_RE: ClassVar[re.Pattern[str]] = re.compile(r"invalid choice: '?([^']+)'? \(choose from (.*)\)")

    def error(self, message: str) -> None:  # type: ignore
        r"""Print help text and error message before terminating CLI session.

        Args:
            message (str): Error message string reported by parser.

        Raises:
            SystemExit: Exits process with status code 2.
        """
        if match := self._INVALID_CHOICE_RE.search(message):
            from difflib import get_close_matches

            invalid = match.group(1)
            choices = [c.strip("'").strip() for c in match.group(2).split(", ")]

            if matches := get_close_matches(invalid, choices):
                message += f"\n    💡 Did you mean '{Colors.info(matches[0])}'?"

        self.print_help(sys.stderr)
        self.exit(2, f"\n{Colors.error('❌ Error:')} {message}\n")


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
        )
        self.verbose = False
        self.formatter_class = self.config.formatter_class
        self.catch_exceptions: tuple[type[Exception], ...] = self.config.catch_exceptions

        self.global_parser = HelpOnErrorParser(add_help=False)
        self.global_parser.add_argument("-V", "--verbose", action="store_true", help="Enable verbose output/progress")

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

        self.subparsers = self.parser.add_subparsers(title=Colors.bold("💬 Commands"), dest="command", required=True)
        self.commands: list[BaseCommand] = []
        self._open_handles: list[IO[Any]] = []
        self.add_commands(*commands)

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

    def run(self, args: list[str] | None = None) -> None:
        r"""Parse arguments and execute bound subcommand handler.

        Args:
            args (list[str] | None): Command line arguments to parse. Defaults to `sys.argv[1:]`.

        Raises:
            SystemExit: If a tracked exception in `catch_exceptions` occurs during execution.
        """
        parsed_args = self.parser.parse_args(args)
        self.verbose = getattr(parsed_args, "verbose", False)

        if hasattr(parsed_args, "func"):
            try:
                parsed_args.func(parsed_args)
            except self.catch_exceptions as exc:
                exit_code = getattr(exc, "exit_code", 1)
                self.exit(str(exc), code=exit_code)
        else:
            self.parser.print_help()

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
            print(msg, file=sys.stderr, **kwargs)

    def progress(self, iterable: Iterable[Any], msg: str) -> Iterable[Any]:
        r"""Wrap iterable to display live progress counter on stderr when verbose mode is active.

        Uses a fast `yield from` pass-through when `verbose` is disabled.

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
