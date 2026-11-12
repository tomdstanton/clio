import dataclasses
import importlib
import importlib.util
import inspect
import json
import os
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from clio.agent import AgentErrorSpec, AgentResult, _detect_agent_mode, serialize_for_json
from clio.cli import Cli, PackageMeta, _has_unquoted_json_flag
from clio.color import Colors
from clio.command import BaseCommand, Command, CommandTarget
from clio.docstr import Arg, DocstringParser, ParameterSpec
from clio.parser import _DEFAULT_FORMATTER, CliError, HelpFormatter, ParseError, ScanPlan, _fuzzy_suggest

_RUNNER_EPILOG = (
    "Examples:\n"
    "  clio path/to/script.py:my_fn --dry-run --json --count 3\n"
    "  clio path/to/script.py --mcp\n"
    "  clio my_pkg.tools:build_index --schema"
)

_RUNNER_POSITIONALS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        name="target",
        annotation=str,
        docstring="Python file or module target (`path/to/script.py[:symbol]` or `pkg.module[:symbol]`).",
        is_optional=False,
        arg_meta=Arg(positional=True, metavar="TARGET"),
    ),
    ParameterSpec(
        name="args",
        annotation=list[str],
        default=(),
        docstring="Arguments forwarded to the target CLI (`--help`, `--json`, `--dry-run`, `--mcp`, ...).",
        is_optional=True,
        arg_meta=Arg(positional=True, metavar="ARGS"),
    ),
)

_RUNNER_SCAN_PLAN: ScanPlan = ScanPlan.compile(_RUNNER_POSITIONALS)


def _identity_dataclass_handler(instance: Any) -> Any:
    r"""Return a hydrated `@dataclass` instance directly when invoked via `:DataclassName`."""
    return instance


@dataclass(frozen=True, slots=True)
class TargetSpec:
    r"""Immutable parsed representation of a `clio` runner target specifier.

    Attributes:
        module_or_path (str): File path (`path/to/script.py`) or dotted module path (`pkg.mod`).
        symbol (str | None): Optional attribute or dotted symbol path after `:` (`my_fn`, `Cls.run`).
        is_file (bool): Whether `module_or_path` refers to a filesystem `.py` script rather than an import path.
    """

    module_or_path: str
    symbol: str | None = None
    is_file: bool = False

    @classmethod
    def parse(cls, raw: str) -> Self:
        r"""Parse a target specifier string (`path/to/script.py[:symbol]` or `pkg.mod[:symbol]`).

        Handles POSIX paths, Windows drive-letter paths (`C:\path\script.py:fn`), and dotted module names.

        Args:
            raw (str): Raw target string from CLI or programmatic caller.

        Returns:
            Self: Parsed `TargetSpec` instance.

        Raises:
            CliError: If `raw` is empty or has a malformed `:` separator.
        """
        cleaned = raw.strip()
        if not cleaned:
            raise CliError("Target specification cannot be empty.", exit_code=2)

        colon_search_start = 2 if (len(cleaned) >= 2 and cleaned[1] == ":" and cleaned[0].isalpha()) else 0
        colon_idx = cleaned.find(":", colon_search_start)

        if colon_idx == -1:
            mod_part = cleaned
            sym_part: str | None = None
        else:
            mod_part = cleaned[:colon_idx].strip()
            raw_sym = cleaned[colon_idx + 1 :].strip()
            if not mod_part:
                raise CliError(
                    f"Invalid target {raw!r}: module or file path before ':' cannot be empty.",
                    exit_code=2,
                )
            if not raw_sym:
                raise CliError(
                    f"Invalid target {raw!r}: symbol name after ':' cannot be empty.",
                    exit_code=2,
                )
            sym_part = raw_sym

        is_file = mod_part.lower().endswith(".py") or "/" in mod_part or "\\" in mod_part or Path(mod_part).is_file()
        return cls(module_or_path=mod_part, symbol=sym_part, is_file=is_file)


def load_target_module(spec: TargetSpec) -> types.ModuleType:
    r"""Load and return the Python module described by `spec` without executing `if __name__ == "__main__":`.

    If `spec` points to a `.py` file inside a package hierarchy containing `__init__.py` files,
    imports it under its canonical dotted package path so relative imports succeed. Standalone
    scripts are loaded via `importlib.util.spec_from_file_location` and registered in `sys.modules`
    prior to execution so `@dataclass` and Python 3.14 `annotationlib` forward references resolve.

    Args:
        spec (TargetSpec): Parsed target specification.

    Returns:
        types.ModuleType: Loaded Python module object.

    Raises:
        CliError: If the target file does not exist or importing raises an exception.
    """
    if spec.is_file:
        file_path = Path(spec.module_or_path)
        if file_path.is_dir() and (file_path / "__init__.py").is_file():
            file_path = file_path / "__init__.py"
        if not file_path.is_file():
            raise CliError(f"Target file not found: {spec.module_or_path!r}", exit_code=2)

        resolved = file_path.resolve()
        pkg_parts: list[str] = []
        if resolved.name == "__init__.py":
            cur = resolved.parent
            while (cur / "__init__.py").is_file():
                pkg_parts.append(cur.name)
                cur = cur.parent
            search_root = cur
            pkg_parts.reverse()
            dotted_name = ".".join(pkg_parts)
        else:
            cur = resolved.parent
            while (cur / "__init__.py").is_file():
                pkg_parts.append(cur.name)
                cur = cur.parent
            search_root = cur
            if pkg_parts:
                pkg_parts.reverse()
                dotted_name = ".".join((*pkg_parts, resolved.stem))
            else:
                dotted_name = ""

        try:
            if dotted_name:
                root_str = str(search_root)
                if root_str not in sys.path:
                    sys.path.insert(0, root_str)
                return importlib.import_module(dotted_name)

            parent_str = str(resolved.parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)

            mtime_ns = resolved.stat().st_mtime_ns
            mod_name = f"_clio_target_{abs(hash((str(resolved), mtime_ns))):x}_{resolved.stem}"
            if (existing := sys.modules.get(mod_name)) is not None:
                return existing

            mod_spec = importlib.util.spec_from_file_location(mod_name, resolved)
            if mod_spec is None or mod_spec.loader is None:
                raise CliError(f"Unable to create module spec for {spec.module_or_path!r}.", exit_code=2)

            module = importlib.util.module_from_spec(mod_spec)
            sys.modules[mod_name] = module
            try:
                mod_spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(mod_name, None)
                raise
            return module
        except CliError:
            raise
        except Exception as exc:
            raise CliError(f"Failed to load target file {spec.module_or_path!r}: {exc}", exit_code=2) from exc

    cwd = os.getcwd()
    if "" not in sys.path and cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        return importlib.import_module(spec.module_or_path)
    except Exception as exc:
        raise CliError(f"Failed to import target module {spec.module_or_path!r}: {exc}", exit_code=2) from exc


def _collect_nested_subcommands(cmd_cls: type[BaseCommand], seen: set[type[Any]]) -> None:
    r"""Collect child command classes declared in `cmd_cls.subcommands` so module discovery avoids duplicating them."""
    raw_subs = cmd_cls.__dict__.get("subcommands") or ()
    for item in raw_subs:
        target_cls = item if isinstance(item, type) else type(item)
        if isinstance(item, type) and target_cls not in seen:
            seen.add(target_cls)
            if issubclass(target_cls, BaseCommand):
                _collect_nested_subcommands(target_cls, seen)


def discover_module_targets(module: types.ModuleType) -> list[CommandTarget]:
    r"""Discover public functions, `BaseCommand`s, and callable `@dataclass`es defined directly in `module`.

    Preserves source definition order, ignores imported symbols from external modules, skips private
    (`_`-prefixed) attributes, and avoids duplicating child commands already nested under a parent
    `CommandGroup` or `Command` in the same module.

    Args:
        module (types.ModuleType): Loaded Python module to scan.

    Returns:
        list[CommandTarget]: Ordered list of discovered command targets suitable for `Cli(commands=...)`.
    """
    mod_name = module.__name__
    nested_children: set[type[Any]] = set()
    for attr_name, obj in module.__dict__.items():
        if attr_name.startswith("_"):
            continue
        if isinstance(obj, type) and obj.__module__ == mod_name and issubclass(obj, BaseCommand):
            _collect_nested_subcommands(obj, nested_children)

    targets: list[CommandTarget] = []
    for attr_name, obj in module.__dict__.items():
        if attr_name.startswith("_"):
            continue
        if isinstance(obj, BaseCommand):
            targets.append(obj)
            continue
        if isinstance(obj, type):
            if obj.__module__ != mod_name or obj in nested_children:
                continue
            if issubclass(obj, BaseCommand):
                if not inspect.isabstract(obj):
                    targets.append(obj)
                continue
            if dataclasses.is_dataclass(obj) and "__call__" in obj.__dict__:
                targets.append(obj)
                continue
            continue
        if callable(obj):
            unwrapped = inspect.unwrap(obj)
            if getattr(unwrapped, "__module__", None) == mod_name:
                targets.append(obj)

    return targets


def _upgrade_existing_cli(
    existing: Cli,
    *,
    prog: str | None = None,
    enable_agent_flags: bool = True,
    agent_mode: bool | None = None,
) -> Cli:
    r"""Ensure an existing `Cli` instance has agent flags enabled when run through the `clio` runner."""
    if enable_agent_flags and not existing.config.enable_agent_flags:
        existing.config = dataclasses.replace(existing.config, enable_agent_flags=True)
    if prog is not None:
        existing.prog = prog
    if agent_mode is not None:
        existing._agent_mode_override = agent_mode
    return existing


def _default_module_prog(spec: TargetSpec) -> str:
    r"""Derive a clean default CLI program name from a file or module `TargetSpec`."""
    if spec.is_file:
        p = Path(spec.module_or_path)
        if p.name == "__init__.py":
            return p.resolve().parent.name or "cli"
        return p.stem or "cli"
    return spec.module_or_path.rsplit(".", 1)[-1] or "cli"


def build_target_cli(
    target: str | TargetSpec,
    *,
    prog: str | None = None,
    enable_agent_flags: bool = True,
    agent_mode: bool | None = None,
) -> Cli:
    r"""Resolve a file or module target specifier into an executable, agent-ready [`Cli`][clio.cli.Cli].

    - When `:symbol` is provided (`script.py:my_fn`, `pkg.mod:MyDataclass`, `pkg.mod:app`), wraps the
      resolved function, `@dataclass`, `Command`, or `Cli` as a ready-to-run `Cli`.
    - When `:symbol` is omitted (`script.py`, `pkg.mod`), reuses an exported `Cli` instance if one exists,
      or auto-discovers all top-level public functions, `Command`s, and callable `@dataclass`es defined
      in that module as lazy subcommands / MCP tools.

    Args:
        target (str | TargetSpec): Target string or parsed `TargetSpec`.
        prog (str | None): Optional override for the generated CLI program name.
        enable_agent_flags (bool): Enable `--json`, `--dry-run`, `--schema`, `--help-md`, `--skill`, and `--mcp`.
        agent_mode (bool | None): Optional override for `CLIO_AGENT=1` detection.

    Returns:
        Cli: Configured `Cli` instance ready for `.run()`, `.run_async()`, or `.serve_mcp()`.

    Raises:
        CliError: If the module/symbol cannot be loaded or contains no runnable targets.
    """
    spec = TargetSpec.parse(target) if isinstance(target, str) else target
    module = load_target_module(spec)

    if spec.symbol is not None:
        cur: Any = module
        traversed = spec.module_or_path
        for part in spec.symbol.split("."):
            try:
                cur = getattr(cur, part)
            except AttributeError as exc:
                public_attrs = [name for name in dir(cur) if not name.startswith("_")]
                raise ParseError(
                    f"Symbol {part!r} not found in {traversed!r}.",
                    suggestions=_fuzzy_suggest(part, public_attrs),
                    command_path=traversed,
                ) from exc
            traversed = f"{traversed}:{part}" if traversed == spec.module_or_path else f"{traversed}.{part}"

        if isinstance(cur, Cli):
            return _upgrade_existing_cli(
                cur,
                prog=prog,
                enable_agent_flags=enable_agent_flags,
                agent_mode=agent_mode,
            )

        if isinstance(cur, BaseCommand):
            return Cli(
                prog=prog or cur.name or spec.symbol.rsplit(".", 1)[-1],
                command=cur,
                catch_exceptions=(Exception,),
                enable_agent_flags=enable_agent_flags,
                agent_mode=agent_mode,
            )

        if isinstance(cur, type):
            if issubclass(cur, Cli):
                return _upgrade_existing_cli(
                    cur(),
                    prog=prog,
                    enable_agent_flags=enable_agent_flags,
                    agent_mode=agent_mode,
                )
            if issubclass(cur, BaseCommand):
                if inspect.isabstract(cur):
                    raise CliError(f"Target command class {cur.__name__!r} is abstract.", exit_code=2)
                return Cli(
                    prog=prog,
                    command=cur,
                    catch_exceptions=(Exception,),
                    enable_agent_flags=enable_agent_flags,
                    agent_mode=agent_mode,
                )
            if dataclasses.is_dataclass(cur):
                dc_handler = None if "__call__" in cur.__dict__ else _identity_dataclass_handler
                dc_cmd = Command.from_dataclass(cur, handler=dc_handler)
                return Cli(
                    prog=prog or dc_cmd.name,
                    command=dc_cmd,
                    catch_exceptions=(Exception,),
                    enable_agent_flags=enable_agent_flags,
                    agent_mode=agent_mode,
                )
            if "__call__" in cur.__dict__:
                try:
                    callable_inst = cur()
                except TypeError as exc:
                    raise CliError(
                        f"Callable class {cur.__name__!r} requires constructor arguments; "
                        f"decorate it with @dataclass or target a specific method.",
                        exit_code=2,
                    ) from exc
                fn_cmd = Command.from_function(
                    callable_inst.__call__,
                    name=cur.__name__.lower(),
                )
                return Cli(
                    prog=prog or fn_cmd.name,
                    command=fn_cmd,
                    catch_exceptions=(Exception,),
                    enable_agent_flags=enable_agent_flags,
                    agent_mode=agent_mode,
                )
            raise CliError(
                f"Target class {cur.__name__!r} is not a Command, @dataclass, or zero-arg callable class.",
                exit_code=2,
            )

        if callable(cur):
            fn_cmd = Command.from_function(cur)
            return Cli(
                prog=prog or fn_cmd.name,
                command=fn_cmd,
                catch_exceptions=(Exception,),
                enable_agent_flags=enable_agent_flags,
                agent_mode=agent_mode,
            )

        raise CliError(
            f"Target '{spec.module_or_path}:{spec.symbol}' resolved to non-callable object {cur!r}.",
            exit_code=2,
        )

    # Whole-module mode: prefer an explicit Cli instance (`cli` / `app` first, else first Cli in module)
    cli_instances: dict[str, Cli] = {
        k: v for k, v in module.__dict__.items() if not k.startswith("_") and isinstance(v, Cli)
    }
    if cli_instances:
        chosen = cli_instances.get("cli") or cli_instances.get("app") or next(iter(cli_instances.values()))
        return _upgrade_existing_cli(
            chosen,
            prog=prog,
            enable_agent_flags=enable_agent_flags,
            agent_mode=agent_mode,
        )

    discovered = discover_module_targets(module)
    if not discovered:
        raise CliError(
            f"No public functions, Commands, or callable @dataclasses found in {spec.module_or_path!r}.",
            exit_code=2,
        )

    mod_desc, _ = DocstringParser.parse_description(module)
    return Cli(
        prog=prog or _default_module_prog(spec),
        description=mod_desc or None,
        commands=discovered,
        catch_exceptions=(Exception,),
        enable_agent_flags=enable_agent_flags,
        agent_mode=agent_mode,
    )


def run_target(
    target: str | TargetSpec,
    args: Sequence[str] | None = None,
    *,
    prog: str | None = None,
    enable_agent_flags: bool = True,
    agent_mode: bool | None = None,
) -> Any:
    r"""Build and synchronously execute a `Cli` for `target` with `args`."""
    cli = build_target_cli(
        target,
        prog=prog,
        enable_agent_flags=enable_agent_flags,
        agent_mode=agent_mode,
    )
    return cli.run([] if args is None else args)


async def run_target_async(
    target: str | TargetSpec,
    args: Sequence[str] | None = None,
    *,
    prog: str | None = None,
    enable_agent_flags: bool = True,
    agent_mode: bool | None = None,
) -> Any:
    r"""Build and asynchronously execute a `Cli` for `target` on an active event loop."""
    cli = build_target_cli(
        target,
        prog=prog,
        enable_agent_flags=enable_agent_flags,
        agent_mode=agent_mode,
    )
    return await cli.run_async([] if args is None else args)


def format_runner_help(formatter: HelpFormatter = _DEFAULT_FORMATTER) -> str:
    r"""Render ANSI help documentation for the `python -m clio` / `clio` target runner."""
    return formatter.format_help(
        "clio",
        description="Zero-edit agent-native CLI and MCP target runner for Python functions, @dataclasses, and modules.",
        epilog=_RUNNER_EPILOG,
        scan_plan=_RUNNER_SCAN_PLAN,
        subcommands=(),
        has_version=True,
    )


def _emit_runner_error(
    err: CliError,
    *,
    command_label: str,
    json_requested: bool,
    formatter: HelpFormatter = _DEFAULT_FORMATTER,
) -> None:
    r"""Emit a runner-level loading or argument error respecting `--json` and `CLIO_AGENT=1`."""
    usage = Colors.strip_ansi(
        formatter.format_usage("clio", positionals=_RUNNER_POSITIONALS, has_options=True, has_subcommands=False)
    ).strip()
    suggestions = err.suggestions if isinstance(err, ParseError) else ()

    if json_requested:
        payload = AgentResult(
            ok=False,
            command=command_label,
            error=AgentErrorSpec(
                message=str(err),
                exit_code=err.exit_code,
                usage=usage,
                suggestions=suggestions,
            ),
        )
        print(payload.to_json())
        sys.exit(err.exit_code)

    if _detect_agent_mode():
        sys.stderr.write(f"{usage}\nclio: error: {err}\n")
        if suggestions:
            sys.stderr.write(f"did you mean: {', '.join(suggestions)}\n")
        sys.exit(err.exit_code)

    styled_usage = formatter.format_usage(
        "clio",
        positionals=_RUNNER_POSITIONALS,
        has_options=True,
        has_subcommands=False,
    ).rstrip("\n")
    hint_text = Colors.muted("Try 'clio --help' for help.")
    if formatter.should_box(agent_mode=False):
        panel = formatter.format_error_box(str(err), suggestions=suggestions)
        sys.stderr.write(f"{styled_usage}\n{hint_text}\n\n{panel}\n")
        sys.exit(err.exit_code)

    sys.stderr.write(f"{styled_usage}\n{hint_text}\n\n{Colors.style('Error:', fg=Colors.RED, bold=True)} {err}\n")
    if suggestions:
        styled = ", ".join(Colors.style(s, fg=Colors.CYAN, bold=True) for s in suggestions)
        sys.stderr.write(f"Did you mean: {styled}?\n")
    sys.exit(err.exit_code)


def main(argv: Sequence[str] | None = None) -> None:
    r"""Entrypoint for `python -m clio`, `uv run -m clio`, and the `clio` console script.

    Returns `None` on success so console-script wrappers (`sys.exit(main())`) exit with status 0
    instead of treating non-integer return values as failure messages on stderr.

    Args:
        argv (Sequence[str] | None): CLI tokens excluding `sys.argv[0]`. Defaults to `sys.argv[1:]`.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        print(format_runner_help(), end="")
        return None

    first = tokens[0]
    if first in ("-h", "--help"):
        print(format_runner_help(), end="")
        sys.exit(0)

    if first in ("-v", "--version"):
        pkg_meta = PackageMeta.resolve("clio")
        ver = (pkg_meta.version if pkg_meta is not None else "") or "0.0.0"
        print(f"clio {ver}")
        sys.exit(0)

    if first.startswith("-"):
        json_req = _has_unquoted_json_flag(tokens, enabled=True)
        err = ParseError(
            f"unrecognized runner option {first!r} before TARGET",
            suggestions=_fuzzy_suggest(first.split("=", 1)[0], ("-h", "--help", "-v", "--version")),
            command_path="clio",
        )
        _emit_runner_error(err, command_label="clio", json_requested=json_req)

    target_raw = first
    target_args = tokens[1:]
    json_req = _has_unquoted_json_flag(target_args, enabled=True)

    try:
        cli = build_target_cli(target_raw, enable_agent_flags=True)
    except CliError as err:
        _emit_runner_error(err, command_label=target_raw, json_requested=json_req)
        raise AssertionError("unreachable") from err  # pragma: no cover

    result = cli.run(target_args)
    if result is not None and not isinstance(result, AgentResult):
        serialized = serialize_for_json(result)
        if isinstance(serialized, (dict, list)):
            print(json.dumps(serialized, indent=2))
        else:
            print(serialized)
    return None
