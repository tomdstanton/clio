import argparse
import dataclasses
import inspect
import types
from abc import ABC, abstractmethod
from annotationlib import ForwardRef, type_repr
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from clio.color import ColorHelpFormatter, Colors
from clio.docstr import DocstringParser, DocstringSpec, ParameterSpec

if TYPE_CHECKING:
    from clio.cli import Cli


def _get_cmd_meta(target_cls: type[Any], attr: str, default: Any = "") -> Any:
    r"""Extract class-level command metadata (`name`, `aliases`, `description`, `help_text`) safely.

    Handles `@dataclass(slots=True)` classes where slotted attributes become `member_descriptor`
    objects and distinguishes `init=True` CLI fields from command metadata.
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


@dataclass(frozen=True, slots=True)
class DataclassBinding:
    r"""Immutable descriptor for constructing a `@dataclass` parameter from parsed CLI flags.

    Attributes:
        param_name (str): Parameter name on `Command.__call__`.
        dc_cls (type[Any]): Target `@dataclass` type to instantiate.
        field_names (tuple[str, ...]): Dataclass `init=True` field names to extract from `Namespace`.
        is_positional_only (bool): Whether the parameter is positional-only in `__call__`.
    """

    param_name: str
    dc_cls: type[Any]
    field_names: tuple[str, ...]
    is_positional_only: bool = False


@dataclass(frozen=True, slots=True)
class CallPlan:
    r"""Immutable, pre-compiled execution dispatch plan for `Command.__call__`.

    Eliminates runtime `inspect.signature` and `inspect.get_annotations` overhead
    when building parsers and invoking commands.

    Attributes:
        has_custom_call (bool): Whether `__call__` was overridden on the command class.
        is_namespace_call (bool): Whether `__call__` expects a single `argparse.Namespace` argument.
        pos_params (tuple[tuple[str, bool], ...]): Positional-only `(name, is_namespace)` descriptors.
        kw_params (tuple[tuple[str, bool], ...]): Keyword-capable `(name, is_namespace)` descriptors.
        dc_bindings (tuple[DataclassBinding, ...]): Nested `@dataclass` parameter bindings.
        self_dc_fields (tuple[str, ...]): Field names when the `Command` class itself is a `@dataclass`.
        has_var_kw (bool): Whether `__call__` accepts `**kwargs`.
        consumed_names (frozenset[str]): Parameter names explicitly bound by parameters or dataclass fields.
        simple_kw_names (tuple[str, ...]): Fast-path parameter names when only standard keyword parameters are used.
    """

    has_custom_call: bool = False
    is_namespace_call: bool = False
    pos_params: tuple[tuple[str, bool], ...] = ()
    kw_params: tuple[tuple[str, bool], ...] = ()
    dc_bindings: tuple[DataclassBinding, ...] = ()
    self_dc_fields: tuple[str, ...] = ()
    has_var_kw: bool = False
    consumed_names: frozenset[str] = frozenset()
    simple_kw_names: tuple[str, ...] = ()

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

        sig, hints = DocstringParser.inspect_callable(call_fn, owner=cmd_cls)

        params = [p for idx, p in enumerate(sig.parameters.values()) if not (idx == 0 and p.name in ("self", "cls"))]
        non_var_params = [
            p for p in params if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        if len(non_var_params) == 1 and not has_var_kw:
            p0 = non_var_params[0]
            ann0 = hints.get(p0.name, p0.annotation)
            if (
                ann0 is argparse.Namespace
                or ann0 in ("argparse.Namespace", "Namespace")
                or (isinstance(ann0, ForwardRef) and ann0.__forward_arg__ in ("argparse.Namespace", "Namespace"))
            ):
                return cls(has_custom_call=True, is_namespace_call=True)
            if p0.name == "args" and ann0 in (inspect.Parameter.empty, Any) and p0.default is inspect.Parameter.empty:
                return cls(has_custom_call=True, is_namespace_call=True)

        pos_list: list[tuple[str, bool]] = []
        kw_list: list[tuple[str, bool]] = []
        dc_list: list[DataclassBinding] = []
        consumed: set[str] = set()

        for param in params:
            raw_ann = hints.get(param.name, param.annotation)
            ann, _, _ = DocstringParser._unwrap_annotation(raw_ann)
            is_ns = (
                ann is argparse.Namespace
                or ann in ("argparse.Namespace", "Namespace")
                or (isinstance(ann, ForwardRef) and ann.__forward_arg__ in ("argparse.Namespace", "Namespace"))
            )

            if isinstance(ann, type) and dataclasses.is_dataclass(ann):
                f_names = tuple(f.name for f in dataclasses.fields(ann) if f.init)
                dc_list.append(
                    DataclassBinding(
                        param_name=param.name,
                        dc_cls=ann,
                        field_names=f_names,
                        is_positional_only=param.kind == inspect.Parameter.POSITIONAL_ONLY,
                    )
                )
                consumed.update(f_names)
                continue

            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                pos_list.append((param.name, is_ns))
                if not is_ns:
                    consumed.add(param.name)
            elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                kw_list.append((param.name, is_ns))
                if not is_ns:
                    consumed.add(param.name)

        self_dc_fields: tuple[str, ...] = ()
        if not non_var_params and not has_var_kw and dataclasses.is_dataclass(cmd_cls):
            self_dc_fields = tuple(f.name for f in dataclasses.fields(cmd_cls) if f.init)
            consumed.update(self_dc_fields)

        pos_tuple = tuple(pos_list)
        kw_tuple = tuple(kw_list)
        dc_tuple = tuple(dc_list)
        simple_kw: tuple[str, ...] = ()
        if (
            kw_tuple
            and not pos_tuple
            and not dc_tuple
            and not self_dc_fields
            and not has_var_kw
            and all(not is_ns for _, is_ns in kw_tuple)
        ):
            simple_kw = tuple(name for name, _ in kw_tuple)

        return cls(
            has_custom_call=True,
            is_namespace_call=False,
            pos_params=pos_tuple,
            kw_params=kw_tuple,
            dc_bindings=dc_tuple,
            self_dc_fields=self_dc_fields,
            has_var_kw=has_var_kw,
            consumed_names=frozenset(consumed),
            simple_kw_names=simple_kw,
        )

    def invoke(self, command: Command, parsed_args: argparse.Namespace) -> None:
        r"""Execute `command.__call__` using the pre-compiled parameter binding plan.

        Args:
            command (Command): Active command instance.
            parsed_args (argparse.Namespace): Parsed CLI arguments namespace.
        """
        raw_values = vars(parsed_args)
        if self.simple_kw_names:
            command.__call__(**{name: raw_values[name] for name in self.simple_kw_names if name in raw_values})
            return

        pos_args: list[Any] = []
        kw_args: dict[str, Any] = {}

        if self.self_dc_fields:
            for field_name in self.self_dc_fields:
                if field_name in raw_values:
                    object.__setattr__(command, field_name, raw_values[field_name])

        for name, is_ns in self.pos_params:
            if is_ns:
                pos_args.append(parsed_args)
            elif name in raw_values:
                pos_args.append(raw_values[name])

        for name, is_ns in self.kw_params:
            if is_ns:
                kw_args[name] = parsed_args
            elif name in raw_values:
                kw_args[name] = raw_values[name]

        for binding in self.dc_bindings:
            dc_kwargs = {fname: raw_values[fname] for fname in binding.field_names if fname in raw_values}
            dc_instance = binding.dc_cls(**dc_kwargs)
            if binding.is_positional_only:
                pos_args.append(dc_instance)
            else:
                kw_args[binding.param_name] = dc_instance

        if self.has_var_kw:
            consumed = self.consumed_names
            for key, val in raw_values.items():
                if key not in consumed and key not in ("func", "command") and not key.endswith("_subcommand"):
                    kw_args[key] = val

        command.__call__(*pos_args, **kw_args)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    r"""Immutable, pre-compiled class metadata and dispatch specification for a command.

    Attributes:
        name (str): Default command trigger name.
        aliases (tuple[str, ...]): Immutable tuple of command aliases.
        description (str): Cleaned command description text.
        help_text (str): Single-line command summary text.
        docstring_spec (DocstringSpec): Cached docstring and parameter introspection result.
        call_plan (CallPlan): Pre-compiled `__call__` dispatch plan.
    """

    name: str
    aliases: tuple[str, ...]
    description: str
    help_text: str
    docstring_spec: DocstringSpec
    call_plan: CallPlan

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
        )


def _format_action_type(action: argparse.Action) -> str:
    r"""Format the type or choices of an `argparse.Action` for Markdown tables using `annotationlib.type_repr`."""
    if action.choices is not None:
        return ", ".join(f"`{c}`" for c in action.choices)
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        return "`bool`"
    if action.type is not None:
        if isinstance(action.type, type):
            return f"`{type_repr(action.type)}`"
        type_name = getattr(action.type, "__name__", str(action.type)).lstrip("_")
        if type_name == "parse_bool":
            type_name = "bool"
        return f"`{type_name}`"
    return "`str`"


def _format_action_default(action: argparse.Action) -> str:
    r"""Format the default value or required status of an `argparse.Action` for Markdown tables."""
    is_positional = not action.option_strings
    if getattr(action, "required", False) or (is_positional and action.nargs not in ("?", "*")):
        return "*required*"
    if action.default is argparse.SUPPRESS or action.default is None:
        return "`None`"
    return f"`{action.default!r}`"


def _escape_md_cell(text: str | None) -> str:
    r"""Sanitize text for placement inside a Markdown table cell."""
    if not text:
        return ""
    cleaned = Colors.strip_ansi(text).strip()
    return " ".join(cleaned.splitlines()).replace("|", r"\|")


def _coerce_subcommand(cmd: BaseCommand | type[Any]) -> BaseCommand:
    r"""Instantiate a `BaseCommand` class or wrap a `@dataclass` type into a `BaseCommand` instance."""
    if isinstance(cmd, BaseCommand):
        return cmd
    if isinstance(cmd, type):
        if issubclass(cmd, BaseCommand):
            return cmd()
        if dataclasses.is_dataclass(cmd):
            return Command.from_dataclass(cmd)
    raise TypeError(f"Expected a BaseCommand, CommandGroup, or @dataclass, got {cmd!r}")


class BaseCommand(ABC):
    r"""Abstract base class providing shared metadata, parser building, and subcommand routing.

    Subclasses should inherit from [`Command`][clio.command.Command] for executable leaf/hybrid
    commands or [`CommandGroup`][clio.command.CommandGroup] for pure subcommand namespaces.

    Attributes:
        name (str): Identifier name used to trigger command on CLI.
        aliases (Sequence[str]): Alternative trigger names or abbreviations for command.
        description (str): Detailed text explanation shown in command help.
        help_text (str): Brief single-line summary shown in parent subparser tables.
        parser (argparse.ArgumentParser | None): Bound argument parser instance.
        subcommands (list[BaseCommand | type[Any]]): Child subcommands (classes, dataclasses, or instances).
        cli (Cli | None): Parent root CLI host application reference.
    """

    _spec: ClassVar[CommandSpec | None] = None

    name: str = ""
    aliases: Sequence[str] = ()
    description: str = ""
    help_text: str = ""
    subcommands: list[BaseCommand | type[Any]] = []

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
        r"""Initialize command instance from pre-compiled `CommandSpec` and register subcommands."""
        spec = type(self).get_spec()
        self.parser: argparse.ArgumentParser | None = None
        self.cli: Cli | None = None

        self.name = self.name or spec.name
        self.aliases: list[str] = list(self.aliases) if self.aliases else list(spec.aliases)
        self.description = self.description or spec.description
        if not self.help_text:
            if spec.help_text and self.description == spec.description:
                self.help_text = spec.help_text
            elif self.description:
                self.help_text = self.description.strip().split("\n", 1)[0]

        cls_subcommands = type(self).__dict__.get("subcommands", ())
        self.subcommands = [_coerce_subcommand(sub) for sub in (cls_subcommands or ())]

        self.register_subcommands()

    def register_subcommands(self) -> None:  # noqa: B027
        r"""Register child subcommand instances into `subcommands` list."""

    def add_subcommand(self, *commands: BaseCommand | type[Any]) -> Self:
        r"""Append one or more child subcommands (classes, dataclasses, or instances) to `self.subcommands`.

        Args:
            *commands (BaseCommand | type[Any]): Subcommands to register.

        Returns:
            Self: The command instance for method chaining.
        """
        for cmd in commands:
            self.subcommands.append(_coerce_subcommand(cmd))
        return self

    def setup_arguments(self) -> None:  # noqa: B027
        r"""Configure argument flags and positional parameters on `self.parser`."""

    def add_arguments_from(
        self,
        target: type[Any] | Callable[..., Any] | Iterable[ParameterSpec],
    ) -> None:
        r"""Populate `self.parser` arguments by introspecting a class, dataclass, callable, or `ParameterSpec` iterable.

        Args:
            target (type[Any] | Callable[..., Any] | Iterable[ParameterSpec]): Target class, callable,
                or parameter specifications to register on `self.parser`.
        """
        if self.parser is not None:
            DocstringParser.populate_parser(self.parser, target)

    def get_shared_parser(self) -> argparse.ArgumentParser | None:
        r"""Return shared parent parser containing options passed to subcommands.

        Returns:
            argparse.ArgumentParser | None: Shared non-help argument parser or `None`.
        """
        return None

    def _bind_execution(self, spec: CommandSpec) -> None:
        r"""Bind command execution defaults and auto-populated parameters onto `self.parser`."""
        assert self.parser is not None
        DocstringParser.backfill_help(self.parser, spec.docstring_spec.parameters)

    def build(
        self,
        subparsers: argparse._SubParsersAction[Any],
        parent_parsers: list[argparse.ArgumentParser] | None = None,
        formatter_class: type[argparse.HelpFormatter] = ColorHelpFormatter,
    ) -> None:
        r"""Wire command parser and subcommands into parent argparse hierarchy.

        Args:
            subparsers (argparse._SubParsersAction): Target subparser action registry.
            parent_parsers (list[argparse.ArgumentParser] | None): Parent shared parsers to inherit.
            formatter_class (type[argparse.HelpFormatter]): Help formatter class for the subparser.
        """
        spec = type(self).get_spec()
        parents = parent_parsers or []

        self.parser = subparsers.add_parser(
            name=self.name,
            aliases=self.aliases,
            description=Colors.bold(self.description),
            help=self.help_text or self.description,
            parents=parents,
            formatter_class=formatter_class,
        )

        # 1. Add specific arguments for this command
        self.setup_arguments()

        # 2. Auto-populate arguments and bind execution handler
        self._bind_execution(spec)

        # Rename the default options group and move it to the bottom of the help menu
        if hasattr(self.parser, "_optionals"):
            self.parser._optionals.title = Colors.bold("🌎 Global options")
            groups = self.parser._action_groups
            if self.parser._optionals in groups:
                groups.append(groups.pop(groups.index(self.parser._optionals)))

        # 3. Process subcommands (if any)
        if self.subcommands:
            is_required = not self._is_executable()
            sub_action = self.parser.add_subparsers(
                title=Colors.bold(f"'{self.name}' subcommands"),
                dest=f"{self.name}_subcommand",
                required=is_required,
            )

            child_parents = parents.copy()
            if shared := self.get_shared_parser():
                DocstringParser.backfill_help(shared, spec.docstring_spec.parameters)
                child_parents.append(shared)

            built_subs: list[BaseCommand | type[Any]] = []
            for raw_cmd in self.subcommands:
                cmd = _coerce_subcommand(raw_cmd)
                cmd.cli = self.cli
                cmd.build(
                    sub_action,
                    parent_parsers=child_parents,
                    formatter_class=formatter_class,
                )
                built_subs.append(cmd)
            self.subcommands = built_subs

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
        parser = self.parser
        if parser is None:
            root = argparse.ArgumentParser(prog=prog_prefix, formatter_class=ColorHelpFormatter)
            sub = root.add_subparsers(dest="command")
            self.build(sub, formatter_class=ColorHelpFormatter)
            parser = self.parser
            if parser is not None:
                parser.prog = full_prog
        assert parser is not None

        hashes = "#" * max(1, heading_level)
        lines: list[str] = [f"{hashes} `{full_prog}`"]

        if self.aliases:
            alias_str = ", ".join(f"`{a}`" for a in self.aliases)
            lines.append(f"\n*Aliases:* {alias_str}")

        if self.description:
            lines.append(f"\n{Colors.strip_ansi(self.description).strip()}")

        usage = Colors.strip_ansi(parser.format_usage()).strip()
        lines.append(f"\n```text\n{usage}\n```")

        positionals: list[argparse.Action] = []
        optionals: list[argparse.Action] = []
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                continue
            if not action.option_strings:
                positionals.append(action)
            else:
                if not include_global_options and any(
                    opt in ("-h", "--help", "-V", "--verbose") for opt in action.option_strings
                ):
                    continue
                optionals.append(action)

        if positionals:
            lines.append("\n**Positional Arguments**\n")
            lines.append("| Argument | Type / Choices | Description |")
            lines.append("| :--- | :--- | :--- |")
            for action in positionals:
                lines.append(f"| `{action.dest}` | {_format_action_type(action)} | {_escape_md_cell(action.help)} |")

        if optionals:
            lines.append("\n**Options**\n")
            lines.append("| Flag | Type / Choices | Default | Description |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for action in optionals:
                flags = ", ".join(f"`{opt}`" for opt in action.option_strings)
                lines.append(
                    f"| {flags} | {_format_action_type(action)} | {_format_action_default(action)} | {_escape_md_cell(action.help)} |"
                )

        if self.subcommands:
            resolved_subs = [_coerce_subcommand(c) for c in self.subcommands]
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


class CommandGroup(BaseCommand):
    r"""Base class for branch commands that group nested subcommands without a direct action handler."""

    def _is_executable(self) -> bool:
        r"""Indicate that a `CommandGroup` requires a child subcommand."""
        return False


class Command(BaseCommand, ABC):
    r"""Strict abstract base class for executable CLI commands.

    Subclasses must implement [`__call__`][clio.command.Command.__call__].
    """

    def _is_executable(self) -> bool:
        r"""Indicate that this command defines an executable `__call__` handler."""
        return True

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        r"""Execute command business logic for parsed arguments.

        Args:
            *args (Any): Positional arguments or `argparse.Namespace`.
            **kwargs (Any): Keyword arguments mapped from parsed CLI options.
        """

    def _invoke_call(self, parsed_args: argparse.Namespace) -> None:
        r"""Dispatch parsed CLI arguments to `self.__call__` via the pre-compiled `CallPlan`."""
        type(self).get_spec().call_plan.invoke(self, parsed_args)

    def _bind_execution(self, spec: CommandSpec) -> None:
        r"""Auto-populate arguments from `__call__` and bind execution handler on `self.parser`."""
        assert self.parser is not None
        plan = spec.call_plan
        if plan.has_custom_call:
            if plan.is_namespace_call:
                DocstringParser.backfill_help(self.parser, spec.docstring_spec.parameters)
                self.parser.set_defaults(func=self.__call__)
            else:
                DocstringParser.populate_parser(self.parser, spec.docstring_spec)
                self.parser.set_defaults(func=self._invoke_call)
        else:
            DocstringParser.backfill_help(self.parser, spec.docstring_spec.parameters)

    @classmethod
    def from_dataclass(
        cls,
        dc_cls: type[Any],
        handler: Callable[[Any], None] | None = None,
        *,
        name: str = "",
        aliases: Sequence[str] = (),
        description: str = "",
        help_text: str = "",
    ) -> DataclassCommand:
        r"""Construct an executable `Command` directly from a `@dataclass` class.

        When the command runs, `clio` instantiates `dc_cls` with the parsed CLI arguments
        and either passes the dataclass instance to `handler(instance)` or invokes `instance()`
        if the dataclass itself is callable.

        Args:
            dc_cls (type[Any]): Target `@dataclass` type defining CLI fields and docstrings.
            handler (Callable[[Any], None] | None): Optional callback invoked with the constructed
                dataclass instance. If omitted, `dc_cls` must implement `__call__`.
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


class DataclassCommand(Command):
    r"""Executable `Command` adapter backed by a `@dataclass` specification."""

    def __init__(
        self,
        dc_cls: type[Any],
        handler: Callable[[Any], None] | None = None,
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

    def _bind_execution(self, spec: CommandSpec) -> None:
        r"""Populate `self.parser` from the wrapped `@dataclass` specs and bind execution."""
        assert self.parser is not None
        DocstringParser.populate_parser(self.parser, self._doc_spec.specs)
        DocstringParser.backfill_help(self.parser, self._doc_spec.parameters)
        self.parser.set_defaults(func=self.__call__)

    def __call__(self, args: argparse.Namespace) -> None:
        r"""Instantiate the `@dataclass` from parsed CLI arguments and execute its handler."""
        raw_values = vars(args)
        dc_kwargs = {fname: raw_values[fname] for fname in self._field_names if fname in raw_values}
        instance = self._dc_cls(**dc_kwargs)
        if self._handler is not None:
            self._handler(instance)
        else:
            instance()
