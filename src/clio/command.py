import dataclasses
import enum
import inspect
import os
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

from clio.agent import ToolSpec
from clio.color import Colors
from clio.docstr import DocstringParser, DocstringSpec, ParameterSpec
from clio.parser import _DEFAULT_FORMATTER, HelpFormatter, ScanPlan, _clone_default

if TYPE_CHECKING:
    from clio.cli import Cli

type CommandTarget = BaseCommand | type[Any] | Callable[..., Any]


def _get_cmd_meta(target_cls: type[Any], attr: str, default: Any = "") -> Any:
    r"""Extract class-level command metadata (`name`, `aliases`, `description`, `help_text`) safely.

    Handles `@dataclass(slots=True)` classes where slotted attributes become `member_descriptor`
    objects and distinguishes `init=True` CLI fields from command metadata without evaluating `__annotate__`.
    """
    if dataclasses.is_dataclass(target_cls):
        for f in dataclasses.fields(target_cls):
            if f.name == attr:
                if f.init:
                    return default
                if f.default is not dataclasses.MISSING:
                    return f.default
                if f.default_factory is not dataclasses.MISSING:
                    return f.default_factory()
                return default

    val = getattr(target_cls, attr, default)
    if isinstance(val, types.MemberDescriptorType):
        return default
    return val


def _format_spec_type(spec: ParameterSpec) -> str:
    r"""Format the type or allowed choices of a `ParameterSpec` for Markdown tables."""
    choices = spec.choices_tuple
    if choices is not None:
        return ", ".join(f"`{c}`" for c in choices)
    return f"`{spec.type_label}`"


def _format_spec_default(spec: ParameterSpec) -> str:
    r"""Format the default value or required status of a `ParameterSpec` for Markdown tables."""
    if spec.is_required:
        return "*required*"
    if not spec.has_default or spec.default is None:
        return "`None`"
    if isinstance(spec.default, enum.Enum):
        return f"`{spec.default.value!r}`"
    return f"`{spec.default!r}`"


def _escape_md_cell(text: str | None) -> str:
    r"""Sanitize text for placement inside a Markdown table cell."""
    if not text:
        return ""
    cleaned = Colors.strip_ansi(text).strip()
    return " ".join(cleaned.splitlines()).replace("|", r"\|")


def _format_spec_description(spec: ParameterSpec) -> str:
    r"""Format a parameter's Markdown description including `[env: ...]` metadata."""
    desc = spec.docstring
    if spec.arg_meta is not None and spec.arg_meta.env:
        env_tag = f"[env: {spec.arg_meta.env}]"
        if env_tag not in desc:
            desc = f"{desc} {env_tag}".strip()
    return _escape_md_cell(desc)


def _coerce_subcommand(cmd: CommandTarget) -> BaseCommand:
    r"""Instantiate a `BaseCommand` class or wrap a `@dataclass` or function into a `BaseCommand` instance."""
    if isinstance(cmd, BaseCommand):
        return cmd
    if isinstance(cmd, type):
        if issubclass(cmd, BaseCommand):
            base_cls: type[BaseCommand] = cmd
            if dataclasses.is_dataclass(cmd):
                inst: BaseCommand = object.__new__(base_cls)
                for f in dataclasses.fields(cmd):
                    if f.default is not dataclasses.MISSING:
                        object.__setattr__(inst, f.name, f.default)
                    elif f.default_factory is not dataclasses.MISSING:
                        object.__setattr__(inst, f.name, f.default_factory())
                BaseCommand.__init__(inst)
                return inst
            return cmd()
        if dataclasses.is_dataclass(cmd):
            return Command.from_dataclass(cmd)
        raise TypeError(f"Expected a BaseCommand, CommandGroup, @dataclass, or function, got {cmd!r}")
    if callable(cmd):
        return Command.from_function(cmd)
    raise TypeError(f"Expected a BaseCommand, CommandGroup, @dataclass, or function, got {cmd!r}")


class _LazyCommandEntry:
    r"""Internal lazy command descriptor that indexes `(name, aliases)` without triggering PEP 649 `__annotate__`."""

    __slots__ = ("_instance", "_raw", "aliases", "name")

    def __init__(self, target: CommandTarget) -> None:
        if isinstance(target, BaseCommand):
            self._raw: CommandTarget = target
            self._instance: BaseCommand | None = target
            self.name: str = target.name
            self.aliases: tuple[str, ...] = tuple(target.aliases)
            return

        if isinstance(target, type):
            if issubclass(target, BaseCommand):
                if inspect.isabstract(target):
                    raise TypeError(f"Can't instantiate abstract command class {target.__name__}")
                self._raw = target
                self._instance = None
                self.name = str(_get_cmd_meta(target, "name", "") or target.__name__.lower())
                self.aliases = tuple(_get_cmd_meta(target, "aliases", ()) or ())
                return
            if dataclasses.is_dataclass(target):
                if "__call__" not in target.__dict__:
                    raise TypeError(
                        f"Dataclass {target.__name__!r} must either define '__call__' or be wrapped with Command.from_dataclass()."
                    )
                self._raw = target
                self._instance = None
                self.name = str(_get_cmd_meta(target, "name", "") or target.__name__.lower())
                self.aliases = tuple(_get_cmd_meta(target, "aliases", ()) or ())
                return
            raise TypeError(f"Expected a BaseCommand, CommandGroup, @dataclass, or function, got {target!r}")

        if callable(target):
            unwrapped = inspect.unwrap(target)
            raw_name = str(getattr(unwrapped, "__name__", "") or getattr(target, "__name__", "") or "command")
            if raw_name == "<lambda>":
                raw_name = "lambda"
            self._raw = target
            self._instance = None
            self.name = raw_name
            self.aliases = (raw_name.replace("_", "-"),) if "_" in raw_name else ()
            return

        raise TypeError(f"Expected a BaseCommand, CommandGroup, @dataclass, or function, got {target!r}")

    def summary(self) -> str:
        r"""Return single-line command summary text without evaluating parameter annotations."""
        if self._instance is not None:
            return self._instance.help_text or self._instance.description
        if isinstance(self._raw, type):
            explicit_help = str(_get_cmd_meta(self._raw, "help_text", ""))
            if explicit_help:
                return explicit_help
            explicit_desc = str(_get_cmd_meta(self._raw, "description", ""))
            if explicit_desc:
                return explicit_desc.strip().split("\n", 1)[0]
        _, doc_help = DocstringParser.parse_description(self._raw)
        return doc_help

    def resolve(self, cli: Cli | None = None) -> BaseCommand:
        r"""Instantiate (if needed) and bind the underlying `BaseCommand`."""
        if self._instance is None:
            self._instance = _coerce_subcommand(self._raw)
        if cli is not None:
            self._instance.cli = cli
        return self._instance


@dataclass(frozen=True, slots=True)
class DataclassBinding:
    r"""Immutable descriptor for constructing a `@dataclass` parameter from parsed CLI flags.

    Attributes:
        param_name (str): Parameter name on `Command.__call__`.
        dc_cls (type[Any]): Target `@dataclass` type to instantiate.
        field_names (tuple[str, ...]): Dataclass `init=True` field names to extract from parsed values.
        is_positional_only (bool): Whether the parameter is positional-only in `__call__`.
    """

    param_name: str
    dc_cls: type[Any]
    field_names: tuple[str, ...]
    is_positional_only: bool = False


@dataclass(frozen=True, slots=True)
class CallPlan:
    r"""Immutable, pre-compiled execution dispatch plan for `Command.__call__`.

    Eliminates runtime `inspect.signature`, `annotationlib`, and `Namespace` overhead
    when invoking synchronous or asynchronous commands.

    Attributes:
        has_custom_call (bool): Whether `__call__` was overridden on the command class.
        is_async (bool): Whether `__call__` is an `async def` coroutine function.
        pos_params (tuple[str, ...]): Positional-only parameter names in declaration order.
        kw_params (tuple[str, ...]): Keyword-capable parameter names.
        dc_bindings (tuple[DataclassBinding, ...]): Nested `@dataclass` parameter bindings.
        self_dc_fields (tuple[str, ...]): Field names when the `Command` class itself is a `@dataclass`.
        has_var_kw (bool): Whether `__call__` accepts `**kwargs`.
        consumed_names (frozenset[str]): Parameter names explicitly bound by parameters or dataclass fields.
        simple_kw_names (tuple[str, ...]): Fast-path parameter names when only standard keyword parameters are used.
    """

    has_custom_call: bool = False
    is_async: bool = False
    pos_params: tuple[str, ...] = ()
    kw_params: tuple[str, ...] = ()
    dc_bindings: tuple[DataclassBinding, ...] = ()
    self_dc_fields: tuple[str, ...] = ()
    has_var_kw: bool = False
    consumed_names: frozenset[str] = frozenset()
    simple_kw_names: tuple[str, ...] = ()

    @classmethod
    def compile_callable(
        cls,
        call_fn: Callable[..., Any],
        *,
        owner: Any = None,
        allow_self_dc: bool = False,
    ) -> Self:
        r"""Compile a stateless `CallPlan` for an arbitrary sync or `async` callable using `annotationlib`."""
        unwrapped = inspect.unwrap(call_fn)
        is_async = inspect.iscoroutinefunction(call_fn) or inspect.iscoroutinefunction(unwrapped)
        eval_owner: Any = owner if owner is not None else unwrapped
        sig, hints = DocstringParser.inspect_callable(call_fn, owner=eval_owner)

        params = [p for idx, p in enumerate(sig.parameters.values()) if not (idx == 0 and p.name in ("self", "cls"))]
        non_var_params = [
            p for p in params if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        pos_list: list[str] = []
        kw_list: list[str] = []
        dc_list: list[DataclassBinding] = []
        consumed: set[str] = set()

        for param in params:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            raw_ann = hints.get(param.name, param.annotation)
            ann, _, _ = DocstringParser._unwrap_annotation(raw_ann, owner=eval_owner)

            if isinstance(ann, type) and dataclasses.is_dataclass(ann):
                f_names = tuple(f.name for f in dataclasses.fields(ann) if f.init)
                is_pos_only = param.kind == inspect.Parameter.POSITIONAL_ONLY
                dc_list.append(
                    DataclassBinding(
                        param_name=param.name,
                        dc_cls=ann,
                        field_names=f_names,
                        is_positional_only=is_pos_only,
                    )
                )
                if is_pos_only:
                    pos_list.append(param.name)
                consumed.update(f_names)
                continue

            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                pos_list.append(param.name)
                consumed.add(param.name)
            elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                kw_list.append(param.name)
                consumed.add(param.name)

        self_dc_fields: tuple[str, ...] = ()
        if (
            allow_self_dc
            and not non_var_params
            and not has_var_kw
            and isinstance(owner, type)
            and dataclasses.is_dataclass(owner)
        ):
            self_dc_fields = tuple(f.name for f in dataclasses.fields(owner) if f.init)
            consumed.update(self_dc_fields)

        pos_tuple = tuple(pos_list)
        kw_tuple = tuple(kw_list)
        dc_tuple = tuple(dc_list)
        simple_kw: tuple[str, ...] = ()
        if kw_tuple and not pos_tuple and not dc_tuple and not self_dc_fields and not has_var_kw:
            simple_kw = kw_tuple

        return cls(
            has_custom_call=True,
            is_async=is_async,
            pos_params=pos_tuple,
            kw_params=kw_tuple,
            dc_bindings=dc_tuple,
            self_dc_fields=self_dc_fields,
            has_var_kw=has_var_kw,
            consumed_names=frozenset(consumed),
            simple_kw_names=simple_kw,
        )

    @classmethod
    def compile(cls, cmd_cls: type[BaseCommand]) -> Self:
        r"""Compile a stateless `CallPlan` from a command class's `__call__` method using `annotationlib`.

        Args:
            cmd_cls (type[BaseCommand]): Target command class to inspect.

        Returns:
            Self: Compiled `CallPlan` instance.
        """
        if not any("__call__" in b.__dict__ and not DocstringParser._is_ignored_base(b) for b in cmd_cls.__mro__):
            return cls(has_custom_call=False)
        call_fn = cmd_cls.__call__
        if getattr(call_fn, "__isabstractmethod__", False):
            return cls(has_custom_call=False)
        return cls.compile_callable(call_fn, owner=cmd_cls, allow_self_dc=True)

    def invoke_callable(self, fn: Callable[..., Any], values: Mapping[str, Any]) -> Any:
        r"""Execute `fn` using the pre-compiled parameter binding plan.

        Args:
            fn (Callable[..., Any]): Target callable to invoke.
            values (Mapping[str, Any]): Coerced CLI argument dictionary.

        Returns:
            Any: Return value (or coroutine when `is_async` is `True`) produced by `fn`.
        """
        if self.simple_kw_names:
            return fn(**{name: values[name] for name in self.simple_kw_names if name in values})

        pos_args: list[Any] = []
        kw_args: dict[str, Any] = {}
        pos_dc_map = {b.param_name: b for b in self.dc_bindings if b.is_positional_only} if self.dc_bindings else {}

        for name in self.pos_params:
            if (dc_b := pos_dc_map.get(name)) is not None:
                dc_kwargs = {fname: values[fname] for fname in dc_b.field_names if fname in values}
                pos_args.append(dc_b.dc_cls(**dc_kwargs))
            elif name in values:
                pos_args.append(values[name])

        for name in self.kw_params:
            if name in values:
                kw_args[name] = values[name]

        for binding in self.dc_bindings:
            if not binding.is_positional_only:
                dc_kwargs = {fname: values[fname] for fname in binding.field_names if fname in values}
                kw_args[binding.param_name] = binding.dc_cls(**dc_kwargs)

        if self.has_var_kw:
            consumed = self.consumed_names
            for key, val in values.items():
                if key not in consumed:
                    kw_args[key] = val

        return fn(*pos_args, **kw_args)

    def invoke(self, command: Command, values: Mapping[str, Any]) -> Any:
        r"""Execute `command.__call__` using the pre-compiled parameter binding plan.

        Args:
            command (Command): Active command instance.
            values (Mapping[str, Any]): Coerced CLI argument dictionary.

        Returns:
            Any: Return value (or coroutine when `is_async` is `True`) produced by `command.__call__`.
        """
        if self.self_dc_fields:
            for field_name in self.self_dc_fields:
                if field_name in values:
                    object.__setattr__(command, field_name, values[field_name])
        return self.invoke_callable(command.__call__, values)

    def resolve_arguments(self, values: Mapping[str, Any]) -> dict[str, Any]:
        r"""Extract resolved command argument bindings from `values` without executing `__call__`.

        Args:
            values (Mapping[str, Any]): Coerced CLI parameter values.

        Returns:
            dict[str, Any]: Mapping of resolved parameter names to bound values.
        """
        if self.simple_kw_names:
            return {name: values[name] for name in self.simple_kw_names if name in values}

        if not self.has_custom_call:
            return dict(values)

        resolved: dict[str, Any] = {}
        for field_name in self.self_dc_fields:
            if field_name in values:
                resolved[field_name] = values[field_name]

        pos_dc_names = {b.param_name for b in self.dc_bindings if b.is_positional_only}
        for name in self.pos_params:
            if name not in pos_dc_names and name in values:
                resolved[name] = values[name]

        for name in self.kw_params:
            if name in values:
                resolved[name] = values[name]

        for binding in self.dc_bindings:
            dc_kwargs = {fname: values[fname] for fname in binding.field_names if fname in values}
            resolved[binding.param_name] = binding.dc_cls(**dc_kwargs)

        if self.has_var_kw:
            consumed = self.consumed_names
            for key, val in values.items():
                if key not in consumed:
                    resolved[key] = val

        return resolved


@dataclass(frozen=True, slots=True)
class CommandSpec:
    r"""Immutable, pre-compiled class metadata, scanner, and dispatch specification for a command.

    Attributes:
        name (str): Default command trigger name.
        aliases (tuple[str, ...]): Immutable tuple of command aliases.
        description (str): Cleaned command description text.
        help_text (str): Single-line command summary text.
        docstring_spec (DocstringSpec): Cached docstring and parameter introspection result.
        call_plan (CallPlan): Pre-compiled `__call__` dispatch plan.
        scan_plan (ScanPlan): Pre-compiled $O(1)$ CLI token scanner.
    """

    name: str
    aliases: tuple[str, ...]
    description: str
    help_text: str
    docstring_spec: DocstringSpec
    call_plan: CallPlan
    scan_plan: ScanPlan

    @classmethod
    def compile(cls, cmd_cls: type[BaseCommand]) -> Self:
        r"""Compile an immutable `CommandSpec` for a `BaseCommand` subclass.

        Args:
            cmd_cls (type[BaseCommand]): Target command subclass.

        Returns:
            Self: Compiled `CommandSpec` instance.
        """
        doc_spec = DocstringParser.inspect(cmd_cls)
        raw_name = str(_get_cmd_meta(cmd_cls, "name", "") or cmd_cls.__name__.lower())
        raw_aliases = tuple(_get_cmd_meta(cmd_cls, "aliases", ()) or ())
        raw_desc = str(_get_cmd_meta(cmd_cls, "description", "") or doc_spec.description)
        raw_help = str(_get_cmd_meta(cmd_cls, "help_text", ""))
        if not raw_help:
            if doc_spec.help_text and raw_desc == doc_spec.description:
                raw_help = doc_spec.help_text
            elif raw_desc:
                raw_help = raw_desc.strip().split("\n", 1)[0]

        return cls(
            name=raw_name,
            aliases=raw_aliases,
            description=raw_desc,
            help_text=raw_help,
            docstring_spec=doc_spec,
            call_plan=CallPlan.compile(cmd_cls),
            scan_plan=ScanPlan.compile(doc_spec.specs),
        )


class BaseCommand(ABC):
    r"""Abstract base class providing shared metadata, lazy subcommand routing, and introspection.

    Subclasses should inherit from [`Command`][clio.command.Command] for executable leaf/hybrid
    commands or [`CommandGroup`][clio.command.CommandGroup] for pure subcommand namespaces.

    Attributes:
        name (str): Identifier name used to trigger command on CLI.
        aliases (Sequence[str]): Alternative trigger names or abbreviations for command.
        description (str): Detailed text explanation shown in command help.
        help_text (str): Brief single-line summary shown in parent subcommand tables.
        subcommands (Sequence[CommandTarget]): Declarative child subcommands (classes, dataclasses, or instances).
        cli (Cli | None): Parent root CLI host application reference.
    """

    _spec: ClassVar[CommandSpec | None] = None

    name: str = ""
    aliases: Sequence[str] = ()
    description: str = ""
    help_text: str = ""
    subcommands: Sequence[CommandTarget] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        r"""Reset `_spec` slot on subclass definition for lazy compilation via `get_spec()`."""
        super().__init_subclass__(**kwargs)
        cls._spec = None

    @classmethod
    def get_spec(cls) -> CommandSpec:
        r"""Return (and lazily compile on first access) the `CommandSpec` for this command class."""
        spec = cls.__dict__.get("_spec")
        if spec is None:
            spec = CommandSpec.compile(cls)
            cls._spec = spec
        return spec

    @abstractmethod
    def _is_executable(self) -> bool:
        r"""Return whether this command class defines an executable action handler."""

    def __init__(self) -> None:
        r"""Initialize command instance from pre-compiled `CommandSpec` and index subcommands lazily."""
        spec = type(self).get_spec()
        self.cli: Cli | None = None

        self.name = self.name or spec.name
        self.aliases: list[str] = list(self.aliases) if self.aliases else list(spec.aliases)
        self.description = self.description or spec.description
        if not self.help_text:
            if spec.help_text and self.description == spec.description:
                self.help_text = spec.help_text
            elif self.description:
                self.help_text = self.description.strip().split("\n", 1)[0]

        self._sub_entries: list[_LazyCommandEntry] = []
        self._sub_route_map: dict[str, _LazyCommandEntry] = {}

        cls_subcommands: Sequence[CommandTarget] = ()
        for base in type(self).__mro__:
            if base is BaseCommand:
                break
            if "subcommands" in base.__dict__:
                cls_subcommands = base.__dict__["subcommands"] or ()
                break

        if cls_subcommands:
            self.add_subcommand(*cls_subcommands)

        self.register_subcommands()

    def register_subcommands(self) -> None:  # noqa: B027
        r"""Hook for subclasses to register child subcommands via [`add_subcommand`][clio.command.BaseCommand.add_subcommand]."""

    def add_subcommand(self, *commands: CommandTarget) -> Self:
        r"""Lazily index one or more child subcommands (classes, dataclasses, or instances) without compiling specs.

        Args:
            *commands (CommandTarget): Subcommands to register.

        Returns:
            Self: The command instance for method chaining.
        """
        for cmd in commands:
            entry = _LazyCommandEntry(cmd)
            self._sub_entries.append(entry)
            self._sub_route_map[entry.name] = entry
            for alias in entry.aliases:
                self._sub_route_map[alias] = entry
        return self

    def has_subcommands(self) -> bool:
        r"""Return `True` if one or more child subcommands are registered on this command."""
        return bool(self._sub_entries)

    def get_subcommands(self) -> tuple[BaseCommand, ...]:
        r"""Materialize and return all registered child subcommand instances."""
        return tuple(entry.resolve(self.cli) for entry in self._sub_entries)

    def _match_subcommand(self, token: str) -> BaseCommand | None:
        r"""Resolve and return the matching child subcommand for `token` in $O(1)$, or `None`."""
        entry = self._sub_route_map.get(token)
        if entry is None:
            return None
        return entry.resolve(self.cli)

    def _subcommand_names(self) -> tuple[str, ...]:
        r"""Return canonical child subcommand names without instantiating them."""
        return tuple(entry.name for entry in self._sub_entries)

    def _subcommand_help_rows(self) -> tuple[tuple[str, Sequence[str], str], ...]:
        r"""Return `(name, aliases, summary)` rows for `--help` without evaluating child `__annotate__`."""
        return tuple((entry.name, entry.aliases, entry.summary()) for entry in self._sub_entries)

    def get_scan_plan(self) -> ScanPlan:
        r"""Return the compiled `ScanPlan` for this command."""
        return type(self).get_spec().scan_plan

    def get_parameter_specs(self) -> tuple[ParameterSpec, ...]:
        r"""Return all `ParameterSpec` items for this command.

        Returns:
            tuple[ParameterSpec, ...]: Ordered tuple of parameter specifications.
        """
        return type(self).get_spec().docstring_spec.specs

    def format_usage(
        self,
        *,
        prog_prefix: str = "",
        formatter: HelpFormatter | None = None,
    ) -> str:
        r"""Format the single-line CLI usage synopsis for this command."""
        fmt = formatter or (self.cli.config.formatter if self.cli is not None else _DEFAULT_FORMATTER)
        full_prog = f"{prog_prefix} {self.name}".strip() if self.name else prog_prefix.strip()
        scan_plan = self.get_scan_plan()
        return fmt.format_usage(
            full_prog,
            positionals=scan_plan.positionals,
            has_options=True,
            has_subcommands=self.has_subcommands(),
        )

    def format_help(
        self,
        *,
        prog_prefix: str = "",
        formatter: HelpFormatter | None = None,
    ) -> str:
        r"""Format complete ANSI help output for this command."""
        fmt = formatter or (self.cli.config.formatter if self.cli is not None else _DEFAULT_FORMATTER)
        enable_agent = self.cli.config.enable_agent_flags if self.cli is not None else False
        full_prog = f"{prog_prefix} {self.name}".strip() if self.name else prog_prefix.strip()
        return fmt.format_help(
            full_prog,
            description=self.description,
            scan_plan=self.get_scan_plan(),
            subcommands=self._subcommand_help_rows(),
            subcommands_title=f"'{self.name}' subcommands",
            has_version=False,
            enable_agent_flags=enable_agent,
            allow_mcp=False,
        )

    def to_markdown(
        self,
        *,
        heading_level: int = 2,
        prog_prefix: str = "",
        include_global_options: bool = False,
    ) -> str:
        r"""Generate zero-dependency GitHub-flavored Markdown documentation for this command and its subcommands.

        Args:
            heading_level (int): Markdown heading depth (`1` for `#`, `2` for `##`, etc.).
            prog_prefix (str): Parent program/command prefix string.
            include_global_options (bool): Whether to include `-h/--help` and `-V/--verbose` in option tables.

        Returns:
            str: Formatted Markdown reference documentation.
        """
        full_prog = f"{prog_prefix} {self.name}".strip()
        scan_plan = self.get_scan_plan()

        hashes = "#" * max(1, heading_level)
        lines: list[str] = [f"{hashes} `{full_prog}`"]

        if self.aliases:
            alias_str = ", ".join(f"`{a}`" for a in self.aliases)
            lines.append(f"\n*Aliases:* {alias_str}")

        if self.description:
            lines.append(f"\n{Colors.strip_ansi(self.description).strip()}")

        usage = Colors.strip_ansi(self.format_usage(prog_prefix=prog_prefix)).strip()
        lines.append(f"\n```text\n{usage}\n```")

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

        if include_global_options:
            if "-h" not in scan_plan.flag_map and "--help" not in scan_plan.flag_map:
                option_rows.append("| `-h`, `--help` | `bool` | `False` | Show this help message and exit |")
            if "-V" not in scan_plan.flag_map and "--verbose" not in scan_plan.flag_map:
                option_rows.append("| `-V`, `--verbose` | `bool` | `False` | Enable verbose output/progress |")

        if option_rows:
            lines.append("\n**Options**\n")
            lines.append("| Flag | Type / Choices | Default | Description |")
            lines.append("| :--- | :--- | :--- | :--- |")
            lines.extend(option_rows)

        if self.has_subcommands():
            resolved_subs = self.get_subcommands()
            lines.append("\n**Subcommands**\n")
            lines.append("| Subcommand | Aliases | Summary |")
            lines.append("| :--- | :--- | :--- |")
            for sub_cmd in resolved_subs:
                aliases_cell = ", ".join(f"`{a}`" for a in sub_cmd.aliases) if sub_cmd.aliases else "—"
                summary = _escape_md_cell(sub_cmd.help_text or sub_cmd.description)
                lines.append(f"| `{sub_cmd.name}` | {aliases_cell} | {summary} |")

            for sub_cmd in resolved_subs:
                lines.append(
                    "\n"
                    + sub_cmd.to_markdown(
                        heading_level=heading_level + 1,
                        prog_prefix=full_prog,
                        include_global_options=include_global_options,
                    )
                )

        return "\n".join(lines)

    def resolve_arguments(self, values: Mapping[str, Any]) -> dict[str, Any]:
        r"""Resolve parsed parameter values into bound command arguments without executing `__call__`.

        Args:
            values (Mapping[str, Any]): Coerced CLI parameter dictionary.

        Returns:
            dict[str, Any]: Dictionary of resolved parameter names and values.
        """
        return type(self).get_spec().call_plan.resolve_arguments(values)

    def _prepare_argument_dict(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        specs = self.get_parameter_specs()
        values: dict[str, Any] = {}

        for spec in specs:
            if spec.name in arguments and arguments[spec.name] is not None:
                values[spec.name] = spec.coerce_value(arguments[spec.name])
            elif spec.name in arguments and arguments[spec.name] is None:
                if spec.is_optional:
                    values[spec.name] = None
                else:
                    raise ValueError(f"Argument '{spec.name}' cannot be null.")
            else:
                meta = spec.arg_meta
                if meta is not None and meta.env and meta.env in os.environ:
                    values[spec.name] = spec._coerce_env_value(os.environ[meta.env])
                elif spec.has_default:
                    values[spec.name] = _clone_default(spec.default)
                elif spec.is_optional:
                    values[spec.name] = None
                else:
                    raise ValueError(f"Missing required argument: '{spec.name}'")

        for extra_key, extra_val in arguments.items():
            if extra_key not in values:
                values[extra_key] = extra_val

        return values

    def invoke_with_arguments(self, arguments: Mapping[str, Any]) -> Any:
        r"""Validate and coerce a dictionary of arguments and execute this command.

        For `async` commands, returns the unawaited coroutine object (use
        [`invoke_with_arguments_async`][clio.command.BaseCommand.invoke_with_arguments_async]
        to await directly).

        Args:
            arguments (Mapping[str, Any]): Input arguments keyed by parameter name.

        Returns:
            Any: Return value (or coroutine) produced by the command callable.

        Raises:
            ValueError: If required arguments are missing or invalid values are supplied.
            TypeError: If called on a non-executable `CommandGroup`.
        """
        if not isinstance(self, Command):
            raise TypeError(f"Command {self.name!r} is not executable.")
        values = self._prepare_argument_dict(arguments)
        return self._invoke_values(values)

    async def invoke_with_arguments_async(self, arguments: Mapping[str, Any]) -> Any:
        r"""Validate, coerce, and asynchronously execute this command (supporting both sync and `async` commands).

        Args:
            arguments (Mapping[str, Any]): Input arguments keyed by parameter name.

        Returns:
            Any: Resolved return value of the command callable.
        """
        result = self.invoke_with_arguments(arguments)
        if inspect.iscoroutine(result):
            return await result
        return result

    def to_json_schema(self) -> dict[str, Any]:
        r"""Generate a JSON Schema representation of this command and any nested subcommands.

        Returns:
            dict[str, Any]: JSON Schema dictionary (`type: "object"`, `properties`, `required`, etc.).
        """
        specs = self.get_parameter_specs()
        properties: dict[str, Any] = {}
        required: list[str] = []

        for spec in specs:
            properties[spec.name] = spec.to_json_schema()
            if spec.is_required:
                required.append(spec.name)

        schema: dict[str, Any] = {
            "type": "object",
            "title": self.name,
            "description": Colors.strip_ansi(self.description or self.help_text).strip(),
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        if self.aliases:
            schema["x-clio-aliases"] = list(self.aliases)
        if self.has_subcommands():
            resolved_subs = self.get_subcommands()
            schema["subcommands"] = {sub.name: sub.to_json_schema() for sub in resolved_subs}
        return schema

    def to_tool(
        self,
        *,
        prefix: str = "",
        command_path: tuple[str, ...] = (),
    ) -> ToolSpec:
        r"""Convert this command into an LLM / MCP [`ToolSpec`][clio.agent.ToolSpec] definition.

        Args:
            prefix (str): Optional name prefix for nested subcommands (e.g. `"branch"`).
            command_path (tuple[str, ...]): Sequence of parent command names leading to this command.

        Returns:
            ToolSpec: Frozen tool specification with `name`, `description`, and `input_schema`.
        """
        specs = self.get_parameter_specs()
        properties: dict[str, Any] = {}
        required: list[str] = []
        for spec in specs:
            properties[spec.name] = spec.to_json_schema()
            if spec.is_required:
                required.append(spec.name)

        raw_tool_name = f"{prefix}_{self.name}" if prefix else self.name
        tool_name = raw_tool_name.strip("_").replace("-", "_").replace(" ", "_")
        full_path = (*command_path, self.name) if command_path else (self.name,)

        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        desc = Colors.strip_ansi(self.description or self.help_text).strip()
        return ToolSpec(
            name=tool_name,
            description=desc,
            input_schema=input_schema,
            command_path=full_path,
        )

    def _iter_tool_bindings(
        self,
        *,
        prefix: str = "",
        command_path: tuple[str, ...] = (),
    ) -> list[tuple[ToolSpec, BaseCommand]]:
        r"""Collect `(ToolSpec, leaf_command)` pairs across this command and any nested subcommands."""
        bindings: list[tuple[ToolSpec, BaseCommand]] = []
        current_prefix = f"{prefix}_{self.name}" if prefix else self.name
        current_path = (*command_path, self.name)

        if self._is_executable():
            bindings.append((self.to_tool(prefix=prefix, command_path=command_path), self))

        for sub_cmd in self.get_subcommands():
            bindings.extend(
                sub_cmd._iter_tool_bindings(
                    prefix=current_prefix,
                    command_path=current_path,
                )
            )
        return bindings

    def to_tools(
        self,
        *,
        prefix: str = "",
        command_path: tuple[str, ...] = (),
    ) -> list[ToolSpec]:
        r"""Flatten all executable commands in this command hierarchy into [`ToolSpec`][clio.agent.ToolSpec] definitions.

        Args:
            prefix (str): Optional name prefix for tool identifiers.
            command_path (tuple[str, ...]): Parent command path tuple.

        Returns:
            list[ToolSpec]: List of flattened tool specifications.
        """
        return [spec for spec, _ in self._iter_tool_bindings(prefix=prefix, command_path=command_path)]

    def to_skill_md(
        self,
        *,
        prog_prefix: str = "",
        enable_agent_flags: bool = False,
    ) -> str:
        r"""Generate an Agent `SKILL.md` document (with YAML frontmatter) for this command.

        Args:
            prog_prefix (str): Optional parent program prefix string.
            enable_agent_flags (bool): Whether to document opt-in agent CLI flags (`--json`, `--dry-run`, etc.).

        Returns:
            str: Complete `SKILL.md` markdown content.
        """
        full_prog = f"{prog_prefix} {self.name}".strip()
        skill_name = full_prog.replace(" ", "-").replace("_", "-").lower()
        summary = Colors.strip_ansi(self.help_text or self.description or f"Run {full_prog} CLI command.").strip()
        summary_line = " ".join(summary.splitlines())

        lines: list[str] = [
            "---",
            f"name: {skill_name}",
            f"description: {summary_line}",
            "---",
            "",
            f"# `{full_prog}` Agent Skill",
            "",
            "## Agent Execution Tips",
            "",
            "- Set `CLIO_AGENT=1` in the environment to receive token-efficient error messages and disable interactive progress spinners.",
        ]
        if enable_agent_flags:
            lines.extend(
                [
                    f"- Pass `--json` (e.g. `{full_prog} --json ...`) to receive a structured JSON envelope (`ok`, `command`, `result`, `messages`, `error`).",
                    "- Pass `--dry-run` to validate CLI arguments and inspect resolved parameter bindings without side effects.",
                    f"- Run `{full_prog} --schema` to inspect the JSON Schema for this command.",
                ]
            )
        lines.append("")
        lines.append(self.to_markdown(heading_level=2, prog_prefix=prog_prefix))
        return "\n".join(lines)


class CommandGroup(BaseCommand):
    r"""Base class for branch commands that group nested subcommands without a direct action handler."""

    @override
    def _is_executable(self) -> bool:
        r"""Indicate that a `CommandGroup` requires a child subcommand."""
        return False


class Command(BaseCommand, ABC):
    r"""Strict abstract base class for executable synchronous or `async` CLI commands.

    Subclasses must implement [`__call__`][clio.command.Command.__call__].
    """

    @override
    def _is_executable(self) -> bool:
        r"""Indicate that this command defines an executable `__call__` handler."""
        return True

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Execute command business logic for parsed arguments.

        Args:
            *args (Any): Positional arguments mapped from CLI parameters.
            **kwargs (Any): Keyword arguments mapped from CLI options.
        """

    def _invoke_values(self, values: Mapping[str, Any]) -> Any:
        r"""Dispatch parsed CLI values to `self.__call__` via the pre-compiled `CallPlan`."""
        return type(self).get_spec().call_plan.invoke(self, values)

    @classmethod
    def run(
        cls,
        args: Sequence[str] | None = None,
        *,
        prog: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        version: str | None = None,
        catch_exceptions: Iterable[type[Exception]] = (),
        formatter: HelpFormatter = _DEFAULT_FORMATTER,
        enable_agent_flags: bool = False,
        agent_mode: bool | None = None,
    ) -> Any:
        r"""Execute this command class directly as a standalone single-command CLI application.

        Args:
            args (Sequence[str] | None): CLI tokens to parse. Defaults to `sys.argv[1:]`.
            prog (str | None): Optional program name override.
            description (str | None): Optional header description override.
            epilog (str | None): Optional footer epilog text.
            version (str | None): Optional version string enabling `-v` / `--version`.
            catch_exceptions (Iterable[type[Exception]]): Domain exception types to trap cleanly.
            formatter (HelpFormatter): Help formatter specification.
            enable_agent_flags (bool): Opt in to built-in agent CLI flags (`--json`, `--dry-run`, etc.).
            agent_mode (bool | None): Override `CLIO_AGENT=1` environment detection.

        Returns:
            Any: Return value of `__call__`, or `AgentResult` when `--json` / `--dry-run` is active.
        """
        from clio.cli import Cli

        return Cli(
            prog=prog or str(_get_cmd_meta(cls, "name", "") or cls.__name__.lower()),
            description=description,
            epilog=epilog,
            version=version,
            catch_exceptions=catch_exceptions,
            formatter=formatter,
            command=cls,
            enable_agent_flags=enable_agent_flags,
            agent_mode=agent_mode,
        ).run(args)

    @classmethod
    async def run_async(
        cls,
        args: Sequence[str] | None = None,
        *,
        prog: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        version: str | None = None,
        catch_exceptions: Iterable[type[Exception]] = (),
        formatter: HelpFormatter = _DEFAULT_FORMATTER,
        enable_agent_flags: bool = False,
        agent_mode: bool | None = None,
    ) -> Any:
        r"""Asynchronously execute this command class as a standalone single-command CLI on an active event loop."""
        from clio.cli import Cli

        return await Cli(
            prog=prog or str(_get_cmd_meta(cls, "name", "") or cls.__name__.lower()),
            description=description,
            epilog=epilog,
            version=version,
            catch_exceptions=catch_exceptions,
            formatter=formatter,
            command=cls,
            enable_agent_flags=enable_agent_flags,
            agent_mode=agent_mode,
        ).run_async(args)

    @classmethod
    def from_dataclass(
        cls,
        dc_cls: type[Any],
        handler: Callable[[Any], Any] | None = None,
        *,
        name: str = "",
        aliases: Sequence[str] = (),
        description: str = "",
        help_text: str = "",
    ) -> DataclassCommand:
        r"""Construct an executable `Command` directly from a `@dataclass` class.

        When the command runs, `clio` instantiates `dc_cls` with the parsed CLI arguments
        and either passes the dataclass instance to `handler(instance)` or invokes `instance()`
        if the dataclass itself is callable (supporting both sync and `async` callables).

        Args:
            dc_cls (type[Any]): Target `@dataclass` type defining CLI fields and docstrings.
            handler (Callable[[Any], Any] | None): Optional sync or `async` callback invoked with the
                constructed dataclass instance. If omitted, `dc_cls` must implement `__call__`.
            name (str): Optional override for command name.
            aliases (Sequence[str]): Optional command aliases.
            description (str): Optional override for command description.
            help_text (str): Optional override for short help summary.

        Returns:
            DataclassCommand: Configured `Command` adapter wrapping `dc_cls`.

        Raises:
            TypeError: If `dc_cls` is not a dataclass or neither `handler` nor `dc_cls.__call__` is provided.
        """
        if not isinstance(dc_cls, type) or not dataclasses.is_dataclass(dc_cls):
            raise TypeError(f"Expected a @dataclass type, got {dc_cls!r}")
        if handler is None and "__call__" not in dc_cls.__dict__:
            raise TypeError(
                f"Dataclass {dc_cls.__name__!r} must either define '__call__' or be paired with a 'handler' function."
            )
        return DataclassCommand(
            dc_cls=dc_cls,
            handler=handler,
            name=name,
            aliases=aliases,
            description=description,
            help_text=help_text,
        )

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str = "",
        aliases: Sequence[str] = (),
        description: str = "",
        help_text: str = "",
    ) -> FunctionCommand:
        r"""Construct an executable `Command` directly from a standalone synchronous or `async` callable.

        Introspects `fn`'s signature, PEP 649 `__annotate__` hints, PEP 695 `type` aliases,
        `Annotated[T, Arg(...)]` metadata, nested `@dataclass` parameters, and Google/NumPy
        docstring into a native `ScanPlan` + `CallPlan` command.

        Args:
            fn (Callable[..., Any]): Target function, coroutine function, or bound method.
            name (str): Optional override for command name (defaults to `fn.__name__`).
            aliases (Sequence[str]): Optional command aliases (defaults to `(fn_name.replace('_', '-'),)`
                when `fn.__name__` contains underscores).
            description (str): Optional override for command description.
            help_text (str): Optional override for short help summary.

        Returns:
            FunctionCommand: Configured `Command` adapter wrapping `fn`.

        Raises:
            TypeError: If `fn` is not a standalone callable function or method.
        """
        return FunctionCommand(
            fn=fn,
            name=name,
            aliases=aliases,
            description=description,
            help_text=help_text,
        )


class DataclassCommand(Command):
    r"""Executable `Command` adapter backed by a `@dataclass` specification."""

    def __init__(
        self,
        dc_cls: type[Any],
        handler: Callable[[Any], Any] | None = None,
        *,
        name: str = "",
        aliases: Sequence[str] = (),
        description: str = "",
        help_text: str = "",
    ) -> None:
        r"""Initialize a `DataclassCommand` from a target `@dataclass` type."""
        doc_spec = DocstringParser.inspect(dc_cls)
        self._dc_cls = dc_cls
        self._handler = handler
        self._doc_spec = doc_spec
        self._scan_plan = ScanPlan.compile(doc_spec.specs)
        self._field_names = tuple(f.name for f in dataclasses.fields(dc_cls) if f.init)

        self.name = str(name or _get_cmd_meta(dc_cls, "name", "") or dc_cls.__name__.lower())
        self.aliases = list(aliases or _get_cmd_meta(dc_cls, "aliases", ()) or ())
        self.description = str(description or _get_cmd_meta(dc_cls, "description", "") or doc_spec.description)
        self.help_text = str(
            help_text
            or _get_cmd_meta(dc_cls, "help_text", "")
            or doc_spec.help_text
            or (self.description.strip().split("\n", 1)[0] if self.description else "")
        )
        super().__init__()

    @override
    def get_scan_plan(self) -> ScanPlan:
        r"""Return the compiled `ScanPlan` for the wrapped `@dataclass`."""
        return self._scan_plan

    @override
    def get_parameter_specs(self) -> tuple[ParameterSpec, ...]:
        r"""Return the introspected `ParameterSpec` items from the wrapped `@dataclass`."""
        return self._doc_spec.specs

    @override
    def resolve_arguments(self, values: Mapping[str, Any]) -> dict[str, Any]:
        r"""Return the bound `@dataclass` field values from `values`."""
        return {fname: values[fname] for fname in self._field_names if fname in values}

    @override
    def _invoke_values(self, values: Mapping[str, Any]) -> Any:
        r"""Instantiate the `@dataclass` from parsed values and execute its handler or `__call__`."""
        return self(**values)

    @override
    def __call__(self, **values: Any) -> Any:
        r"""Instantiate the `@dataclass` from keyword values and execute its handler."""
        dc_kwargs = {fname: values[fname] for fname in self._field_names if fname in values}
        instance = self._dc_cls(**dc_kwargs)
        if self._handler is not None:
            return self._handler(instance)
        return instance()


class FunctionCommand(Command):
    r"""Executable `Command` adapter backed by a standalone synchronous or `async` callable."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str = "",
        aliases: Sequence[str] = (),
        description: str = "",
        help_text: str = "",
    ) -> None:
        r"""Initialize a `FunctionCommand` from a target callable."""
        if not callable(fn) or isinstance(fn, (type, BaseCommand)):
            raise TypeError(f"Expected a standalone callable function or method, got {fn!r}")

        unwrapped = inspect.unwrap(fn)
        raw_name = str(getattr(unwrapped, "__name__", "") or getattr(fn, "__name__", "") or "command")
        if raw_name == "<lambda>":
            raw_name = "lambda"

        owner: Any = (
            fn.__self__
            if inspect.ismethod(fn) and isinstance(fn.__self__, type)
            else (type(fn.__self__) if inspect.ismethod(fn) else unwrapped)
        )
        doc_spec = DocstringParser.inspect(fn)
        self._fn = fn
        self._doc_spec = doc_spec
        self._scan_plan = ScanPlan.compile(doc_spec.specs)
        self._call_plan = CallPlan.compile_callable(fn, owner=owner)

        default_aliases: tuple[str, ...] = (
            (raw_name.replace("_", "-"),) if not name and not aliases and "_" in raw_name else ()
        )
        self.name = str(name or raw_name)
        self.aliases = list(aliases or default_aliases)
        self.description = str(description or doc_spec.description)
        self.help_text = str(
            help_text or doc_spec.help_text or (self.description.strip().split("\n", 1)[0] if self.description else "")
        )
        super().__init__()

    @override
    def get_scan_plan(self) -> ScanPlan:
        r"""Return the compiled `ScanPlan` for the wrapped callable."""
        return self._scan_plan

    @override
    def get_parameter_specs(self) -> tuple[ParameterSpec, ...]:
        r"""Return the introspected `ParameterSpec` items from the wrapped callable."""
        return self._doc_spec.specs

    @override
    def resolve_arguments(self, values: Mapping[str, Any]) -> dict[str, Any]:
        r"""Return resolved parameter and nested `@dataclass` bindings from `values`."""
        return self._call_plan.resolve_arguments(values)

    @override
    def _invoke_values(self, values: Mapping[str, Any]) -> Any:
        r"""Dispatch parsed CLI values to the wrapped callable via the pre-compiled `CallPlan`."""
        return self._call_plan.invoke_callable(self._fn, values)

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Forward direct Python calls to the wrapped callable."""
        return self._fn(*args, **kwargs)
