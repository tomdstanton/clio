import annotationlib
import dataclasses
import enum
import inspect
import os
import re
import types
from abc import ABC
from annotationlib import Format, ForwardRef, type_repr
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from inspect import Parameter, Signature, _ParameterKind
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAliasType, Union, get_args, get_origin

from clio.agent import serialize_for_json


def _parse_bool(value: str) -> bool:
    r"""Parse a CLI or environment string token into a boolean value.

    Args:
        value (str): Raw string token from the command line or environment.

    Returns:
        bool: Parsed boolean value.

    Raises:
        ValueError: If `value` cannot be interpreted as a boolean.
    """
    match value.strip().lower():
        case "1" | "true" | "t" | "yes" | "y" | "on":
            return True
        case "0" | "false" | "f" | "no" | "n" | "off":
            return False
        case _:
            raise ValueError(f"invalid boolean value: {value!r}")


def _substitute_type_params(tp_expr: Any, param_map: Mapping[Any, Any]) -> Any:
    r"""Substitute PEP 695 generic type parameters inside an alias value expression."""
    if not param_map:
        return tp_expr
    if tp_expr in param_map:
        return param_map[tp_expr]

    origin = get_origin(tp_expr)
    if origin is None:
        return tp_expr

    args = get_args(tp_expr)
    if not args:
        return tp_expr

    if origin is Annotated:
        base_sub = _substitute_type_params(args[0], param_map)
        return Annotated[(base_sub, *args[1:])]  # type: ignore[valid-type]

    if origin in (types.UnionType, Union):
        sub_args = [_substitute_type_params(a, param_map) for a in args]
        acc = sub_args[0]
        for nxt in sub_args[1:]:
            acc = acc | nxt
        return acc

    sub_tuple = tuple(Ellipsis if a is Ellipsis else _substitute_type_params(a, param_map) for a in args)
    try:
        return origin[sub_tuple[0] if len(sub_tuple) == 1 else sub_tuple]
    except TypeError:
        return tp_expr


def _eval_forward_ref(ref: ForwardRef, owner: Any = None) -> Any:
    r"""Evaluate a Python 3.14 `ForwardRef` using `Format.FORWARDREF` without raising `NameError`."""
    try:
        return ref.evaluate(owner=owner, format=Format.FORWARDREF)
    except NameError, AttributeError, TypeError, SyntaxError:
        return ref


def _coerce_scalar(raw: Any, target_ann: Any, param_name: str) -> Any:
    r"""Coerce a single scalar value according to `target_ann`."""
    if isinstance(target_ann, TypeAliasType):
        target_ann, _, _ = DocstringParser._unwrap_annotation(target_ann)

    origin = get_origin(target_ann)
    if origin is Annotated:
        target_ann = get_args(target_ann)[0]
        origin = get_origin(target_ann)

    if target_ann is bool:
        return raw if isinstance(raw, bool) else _parse_bool(str(raw))

    if origin is Literal:
        choices = list(get_args(target_ann))
        coerced = raw
        if choices:
            choice_types = {type(c) for c in choices}
            if len(choice_types) == 1:
                only_type = next(iter(choice_types))
                if only_type is bool:
                    coerced = raw if isinstance(raw, bool) else _parse_bool(str(raw))
                elif not isinstance(raw, only_type):
                    coerced = only_type(raw)
            elif isinstance(raw, str):
                for c in choices:
                    if str(c) == raw:
                        coerced = c
                        break
        if coerced not in choices:
            raise ValueError(f"Invalid choice {raw!r} for '{param_name}' (choose from {choices})")
        return coerced

    if isinstance(target_ann, type) and issubclass(target_ann, enum.Enum):
        if isinstance(raw, target_ann):
            return raw
        for member in target_ann:
            if raw == member.value or str(raw) == str(member.value):
                return member
        if isinstance(raw, str):
            for member in target_ann:
                if member.name == raw or member.name.lower() == raw.lower():
                    return member
        valid = [m.value for m in target_ann]
        raise ValueError(f"Invalid choice {raw!r} for '{param_name}' (choose from {valid})")

    if (
        target_ann not in (inspect.Parameter.empty, Any, object)
        and origin is None
        and not isinstance(target_ann, ForwardRef)
        and callable(target_ann)
    ):
        if isinstance(target_ann, type) and isinstance(raw, target_ann):
            return raw
        return target_ann(raw)

    return raw


@dataclass(frozen=True, slots=True)
class Arg:
    r"""Immutable CLI parameter metadata for `typing.Annotated[T, Arg(...)]`.

    Enables declarative configuration of short flags, help descriptions, metavars,
    explicit choices, environment variable fallbacks, and positional vs. option
    overrides directly inside Python 3.14 type annotations inspected by `annotationlib`.

    Attributes:
        short (str | Sequence[str]): Short flag(s) or custom option aliases (e.g. `"-w"` or `("-w", "--num-workers")`).
        help (str): Explicit help description (overrides docstring `Args:` entry when non-empty).
        metavar (str | None): Custom placeholder name in usage and help output.
        choices (Sequence[Any] | None): Explicit allowed argument values.
        env (str | None): Environment variable name used as a fallback default if set in `os.environ`.
        positional (bool | None): Force positional (`True`) or `--flag` option (`False`) regardless of default value.
    """

    short: str | Sequence[str] = ()
    help: str = ""
    metavar: str | None = None
    choices: Sequence[Any] | None = None
    env: str | None = None
    positional: bool | None = None

    @property
    def short_flags(self) -> tuple[str, ...]:
        r"""Normalized tuple of short or custom option flags."""
        if isinstance(self.short, str):
            return (self.short,) if self.short else ()
        return tuple(flag for flag in self.short if flag)

    def merge(self, other: Self) -> Self:
        r"""Merge two `Arg` specifications, allowing `other` to override non-empty fields.

        Args:
            other (Self): Higher-priority `Arg` metadata specification.

        Returns:
            Self: Combined `Arg` instance.
        """
        merged_short = (*self.short_flags, *(f for f in other.short_flags if f not in self.short_flags))
        return type(self)(
            short=merged_short,
            help=other.help or self.help,
            metavar=other.metavar if other.metavar is not None else self.metavar,
            choices=other.choices if other.choices is not None else self.choices,
            env=other.env if other.env is not None else self.env,
            positional=other.positional if other.positional is not None else self.positional,
        )


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    r"""Immutable introspected metadata for a single callable, `@dataclass`, or `__init__` parameter.

    Attributes:
        name (str): Parameter identifier name.
        annotation (Any): Resolved type annotation (unwrapped from `Annotated`, `TypeAliasType`, and optional union).
        default (Any): Default parameter value, or `inspect.Parameter.empty`.
        docstring (str): Resolved help description from `Arg.help`, field metadata, or docstring.
        is_optional (bool): Whether parameter accepts `None` or has a default value.
        kind (inspect._ParameterKind): Parameter binding kind (positional, keyword-only, etc.).
        arg_meta (Arg | None): Optional `Annotated[T, Arg(...)]` CLI metadata descriptor.
    """

    name: str
    annotation: Any
    default: Any = Parameter.empty
    docstring: str = ""
    is_optional: bool = False
    kind: _ParameterKind = Parameter.POSITIONAL_OR_KEYWORD
    arg_meta: Arg | None = None

    @property
    def has_default(self) -> bool:
        r"""Whether the parameter defines a default value."""
        return self.default is not inspect.Parameter.empty

    @property
    def is_positional(self) -> bool:
        r"""Whether the parameter maps to a positional CLI argument."""
        if self.arg_meta is not None:
            if self.arg_meta.positional is not None:
                return self.arg_meta.positional
            if self.arg_meta.short_flags:
                return False
        return not self.has_default and self.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )

    @property
    def flag_name(self) -> str:
        r"""Primary CLI argument or `--kebab-case` option flag name."""
        return self.name if self.is_positional else f"--{self.name.replace('_', '-')}"

    @property
    def option_flags(self) -> tuple[str, ...]:
        r"""All CLI flag strings (including short flags from `Arg.short`)."""
        if self.is_positional:
            return (self.name,)
        primary = self.flag_name
        if self.arg_meta is None or not self.arg_meta.short_flags:
            return (primary,)
        flags = list(self.arg_meta.short_flags)
        if primary not in flags:
            flags.append(primary)
        return tuple(flags)

    @property
    def negated_flag(self) -> str | None:
        r"""Negated `--no-<name>` option flag for non-positional boolean parameters, or `None`."""
        if self.annotation is bool and not self.is_positional:
            return f"--no-{self.name.replace('_', '-')}"
        return None

    @property
    def show_negated_in_help(self) -> bool:
        r"""Whether `--no-<name>` should be displayed alongside `--<name>` in help output."""
        return self.negated_flag is not None and (not self.has_default or self.default is not False)

    @property
    def type_label(self) -> str:
        r"""Human-readable representation of the parameter's annotation via `annotationlib.type_repr`."""
        if self.annotation is inspect.Parameter.empty:
            return "str"
        if isinstance(self.annotation, ForwardRef):
            return self.annotation.__forward_arg__
        return type_repr(self.annotation)

    @property
    def choices_tuple(self) -> tuple[Any, ...] | None:
        r"""Tuple of allowed choice values if constrained by `Arg.choices`, `Literal`, or `Enum`."""
        if self.arg_meta is not None and self.arg_meta.choices is not None:
            return tuple(self.arg_meta.choices)

        ann = self.annotation
        origin = get_origin(ann)
        if origin is Literal:
            return tuple(get_args(ann))
        if isinstance(ann, type) and issubclass(ann, enum.Enum):
            return tuple(member.value for member in ann)

        if origin in (list, set, Sequence, Iterable, AbstractSet) or (
            origin is tuple and (not get_args(ann) or Ellipsis in get_args(ann))
        ):
            item_args = get_args(ann)
            if item_args:
                item_ann = item_args[0]
                if get_origin(item_ann) is Literal:
                    return tuple(get_args(item_ann))
                if isinstance(item_ann, type) and issubclass(item_ann, enum.Enum):
                    return tuple(member.value for member in item_ann)

        return None

    @property
    def is_variadic_collection(self) -> bool:
        r"""Whether the parameter accepts a variable-length sequence of CLI tokens."""
        ann = self.annotation
        origin = get_origin(ann)
        if origin in (list, set, Sequence, Iterable, AbstractSet) or ann in (list, set):
            return True
        if ann is tuple:
            return True
        if origin is tuple:
            args = get_args(ann)
            return not args or (len(args) == 2 and args[1] is Ellipsis)
        return False

    @property
    def fixed_tuple_types(self) -> tuple[Any, ...] | None:
        r"""Element types when `annotation` is a fixed-length `tuple[T1, T2, ...]`, else `None`."""
        ann = self.annotation
        if get_origin(ann) is tuple:
            args = get_args(ann)
            if args and not (len(args) == 2 and args[1] is Ellipsis):
                if args == ((),):
                    return ()
                return tuple(args)
        return None

    @property
    def token_arity(self) -> tuple[int, int | None]:
        r"""Return `(min_tokens, max_tokens)` consumed per occurrence (`None` max means unbounded)."""
        if self.annotation is bool and not self.is_positional:
            return (0, 0)
        fixed = self.fixed_tuple_types
        if fixed is not None:
            n = len(fixed)
            return (n, n)
        if self.is_variadic_collection:
            min_tok = 0 if (self.has_default or not self.is_required) else 1
            return (min_tok, None)
        return (1, 1)

    @property
    def metavar_label(self) -> str:
        r"""Placeholder label rendered in usage and option help columns."""
        if self.arg_meta is not None and self.arg_meta.metavar is not None:
            return self.arg_meta.metavar
        choices = self.choices_tuple
        if choices is not None:
            return "{" + ",".join(str(c) for c in choices) + "}"
        fixed = self.fixed_tuple_types
        if fixed is not None and fixed:
            base = self.name.upper()
            return " ".join(f"{base}{i + 1}" for i in range(len(fixed)))
        return self.name.upper()

    @property
    def is_required(self) -> bool:
        r"""Whether this parameter is required when invoking the command."""
        if self.has_default:
            return False
        if self.arg_meta is not None and self.arg_meta.env and self.arg_meta.env in os.environ:
            return False
        return True

    def _coerce_env_value(self, raw_env: str) -> Any:
        r"""Convert a raw environment variable string according to `self.annotation`."""
        return self.coerce_value(raw_env)

    @classmethod
    def _annotation_to_json_schema(cls, ann: Any) -> dict[str, Any]:
        r"""Map a resolved Python type annotation to a JSON Schema Draft 2020-12 property fragment."""
        if isinstance(ann, TypeAliasType) or isinstance(get_origin(ann), TypeAliasType):
            ann, _, _ = DocstringParser._unwrap_annotation(ann)

        if ann is inspect.Parameter.empty or ann in (Any, object) or isinstance(ann, ForwardRef):
            return {"type": "string"}

        origin = get_origin(ann)
        if origin is Annotated:
            return cls._annotation_to_json_schema(get_args(ann)[0])

        if origin in (types.UnionType, Union):
            non_none = [a for a in get_args(ann) if a is not type(None)]
            if len(non_none) == 1:
                return cls._annotation_to_json_schema(non_none[0])
            return {"anyOf": [cls._annotation_to_json_schema(a) for a in non_none]}

        if origin is Literal:
            choices = list(get_args(ann))
            schema: dict[str, Any] = {"enum": [serialize_for_json(c) for c in choices]}
            choice_types = {type(c) for c in choices}
            if len(choice_types) == 1:
                only_type = next(iter(choice_types))
                type_schema = cls._annotation_to_json_schema(only_type)
                if "type" in type_schema:
                    schema["type"] = type_schema["type"]
            return schema

        if origin is tuple:
            args = get_args(ann)
            if args and not (len(args) == 2 and args[1] is Ellipsis):
                if args == ((),):
                    return {"type": "array", "maxItems": 0}
                prefix = [cls._annotation_to_json_schema(a) for a in args]
                return {
                    "type": "array",
                    "prefixItems": prefix,
                    "minItems": len(prefix),
                    "maxItems": len(prefix),
                }
            items_schema = cls._annotation_to_json_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": items_schema}

        if origin in (list, set, Sequence, Iterable, AbstractSet) or ann in (list, tuple, set):
            item_args = get_args(ann)
            items_schema = cls._annotation_to_json_schema(item_args[0]) if item_args else {"type": "string"}
            return {"type": "array", "items": items_schema}

        if origin in (dict, Mapping) or ann in (dict, Mapping):
            return {"type": "object"}

        match ann:
            case _ if ann is bool:
                return {"type": "boolean"}
            case _ if ann is int:
                return {"type": "integer"}
            case _ if ann is float:
                return {"type": "number"}
            case _ if ann is str:
                return {"type": "string"}
            case type() if issubclass(ann, os.PathLike):
                return {"type": "string", "format": "path"}
            case type() if issubclass(ann, enum.Enum):
                return {"type": "string", "enum": [serialize_for_json(e.value) for e in ann]}
            case _:
                return {"type": "string"}

    def to_json_schema(self) -> dict[str, Any]:
        r"""Compile parameter specification into a JSON Schema property definition.

        Includes `description`, `default`, `enum` choices, and `x-clio-*` CLI binding metadata.

        Returns:
            dict[str, Any]: JSON Schema property dictionary for this parameter.
        """
        schema = self._annotation_to_json_schema(self.annotation)
        meta = self.arg_meta

        if meta is not None and meta.choices is not None:
            schema["enum"] = [serialize_for_json(c) for c in meta.choices]

        if self.docstring:
            schema["description"] = self.docstring

        if self.has_default and self.default is not None:
            schema["default"] = serialize_for_json(self.default)

        if self.is_positional:
            schema["x-clio-positional"] = True
        else:
            schema["x-clio-flags"] = list(self.option_flags)

        if meta is not None and meta.env:
            schema["x-clio-env"] = meta.env

        return schema

    def coerce_value(self, value: Any) -> Any:
        r"""Validate and coerce a raw CLI token, env string, JSON, or Python value into `self.annotation`.

        Args:
            value (Any): Raw input value or sequence of tokens.

        Returns:
            Any: Coerced Python value matching `self.annotation`.

        Raises:
            ValueError: If `value` violates allowed choices, arity, or type conversion.
        """
        if value is None:
            return None

        ann = self.annotation
        origin = get_origin(ann)
        meta = self.arg_meta

        explicit_choices: list[Any] | None = (
            list(meta.choices) if meta is not None and meta.choices is not None else None
        )

        fixed = self.fixed_tuple_types
        if fixed is not None:
            raw_seq = [p.strip() for p in value.split(",") if p.strip()] if isinstance(value, str) else list(value)
            if len(raw_seq) != len(fixed):
                raise ValueError(f"Expected {len(fixed)} values for '{self.name}', got {len(raw_seq)}")
            coerced_tuple = tuple(
                _coerce_scalar(raw_item, elem_ann, self.name) for raw_item, elem_ann in zip(raw_seq, fixed, strict=True)
            )
            if explicit_choices is not None:
                for item in coerced_tuple:
                    if item not in explicit_choices:
                        raise ValueError(f"Invalid choice {item!r} for '{self.name}' (choose from {explicit_choices})")
            return coerced_tuple

        if self.is_variadic_collection:
            raw_items = [p.strip() for p in value.split(",") if p.strip()] if isinstance(value, str) else list(value)
            item_args = get_args(ann)
            if item_args:
                item_ann = item_args[0]
                coerced_items = [_coerce_scalar(item, item_ann, self.name) for item in raw_items]
            else:
                coerced_items = raw_items

            if explicit_choices is not None:
                for item in coerced_items:
                    if item not in explicit_choices:
                        raise ValueError(f"Invalid choice {item!r} for '{self.name}' (choose from {explicit_choices})")

            if origin is tuple or ann is tuple:
                return tuple(coerced_items)
            if origin in (set, AbstractSet) or ann is set:
                return set(coerced_items)
            return list(coerced_items)

        coerced = _coerce_scalar(value, ann, self.name)
        if explicit_choices is not None and coerced not in explicit_choices:
            if isinstance(coerced, enum.Enum) and coerced.value in explicit_choices:
                return coerced
            raise ValueError(f"Invalid choice {coerced!r} for '{self.name}' (choose from {explicit_choices})")
        return coerced


@dataclass(frozen=True, slots=True)
class DocstringSpec:
    r"""Immutable, cached introspection result for a class or callable.

    Attributes:
        description (str): Cleaned multiline prose description with metadata sections stripped.
        help_text (str): Single-line summary extracted from the first line of `description`.
        parameters (Mapping[str, str]): Read-only mapping of parameter names to docstring descriptions.
        specs (tuple[ParameterSpec, ...]): Ordered tuple of introspected parameter specifications.
    """

    description: str = ""
    help_text: str = ""
    parameters: Mapping[str, str] = field(default_factory=dict)
    specs: tuple[ParameterSpec, ...] = ()


class DocstringParser:
    r"""Stateless `annotationlib`-powered signature and docstring extractor with class-level caching."""

    _CACHE: ClassVar[dict[Any, dict[str, str]]] = {}
    _DESC_CACHE: ClassVar[dict[Any, tuple[str, str]]] = {}
    _SPEC_CACHE: ClassVar[dict[Any, tuple[ParameterSpec, ...]]] = {}
    _DOC_SPEC_CACHE: ClassVar[dict[Any, DocstringSpec]] = {}
    _SIG_CACHE: ClassVar[dict[Any, tuple[Signature, dict[str, Any]]]] = {}

    _SECTION_SPLIT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:^|\n)[ \t]*(?:"
        r"(?:Args|Arguments|Parameters|Keyword Args|Keyword Arguments|Returns|Yields|Raises|Attributes|Note|Notes|Example|Examples|See Also|References|Warnings):[ \t]*(?:\n|$)"
        r"|"
        r"(?:Parameters|Returns|Yields|Raises|Attributes|See Also|Notes|References|Examples|Warnings)[ \t]*\n[ \t]*-+(?:\n|$)"
        r")",
        re.DOTALL,
    )
    _NUMPY_SECTION_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:Parameters|Attributes)\s*\n\s*-+\s*\n(?P<content>.*?)(?=\n\s*(?:Parameters|Returns|Yields|Attributes|Raises|See Also|Notes|References|Examples|Warnings)\s*\n\s*-+|$)",
        re.DOTALL,
    )
    _NUMPY_ITEM_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<name>\w+)\s*(?::\s*(?P<type>[^\n]+))?\n(?P<desc>(?:[ \t]+[^\n]*\n?)+)",
        re.MULTILINE,
    )
    _GOOGLE_SECTION_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:Args|Arguments|Parameters|Keyword Args|Keyword Arguments|Attributes):\s*\n(?P<content>.*?)(?=\n\s*(?:Args|Arguments|Parameters|Keyword Args|Keyword Arguments|Returns|Yields|Raises|Attributes|Note|Notes|Example|Examples|See Also|References|Warnings):\s*\n|$)",
        re.DOTALL,
    )
    _GOOGLE_ITEM_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<indent>[ \t]+)(?P<name>\*{0,2}\w+)(?:\s*\([^\n:]*\))?:\s*(?P<first_line>[^\n]*)(?P<cont>(?:\n(?P=indent)[ \t]+[^\n]*)*)",
        re.MULTILINE,
    )

    @classmethod
    def clear_cache(cls) -> None:
        r"""Clear all cached docstring and `annotationlib` parameter specifications."""
        cls._CACHE.clear()
        cls._DESC_CACHE.clear()
        cls._SPEC_CACHE.clear()
        cls._DOC_SPEC_CACHE.clear()
        cls._SIG_CACHE.clear()

    @staticmethod
    def _is_ignored_base(base: type[Any]) -> bool:
        r"""Check whether a base class in MRO should be ignored during docstring resolution."""
        if base in (object, ABC):
            return True
        return base.__name__ in ("BaseCommand", "Command", "CommandGroup", "DataclassCommand") and (
            base.__module__.startswith("clio")
        )

    @staticmethod
    def _cache_key(target: Any) -> Any:
        r"""Compute a hashable cache key for a target class, method, or callable."""
        if inspect.ismethod(target):
            return (target.__self__ if isinstance(target.__self__, type) else type(target.__self__), target.__func__)
        try:
            hash(target)
        except TypeError:
            return None
        return target

    @classmethod
    def resolve_annotations(
        cls,
        target: Any,
        *,
        owner: type[Any] | None = None,
    ) -> dict[str, Any]:
        r"""Resolve annotations on `target` using Python 3.14 `annotationlib` with `Format.FORWARDREF`.

        Evaluates PEP 649 `__annotate__` functions directly (and stringified annotations via `eval_str=True`),
        evaluating any `ForwardRef`s with `Format.FORWARDREF` so unresolvable names stay safe `ForwardRef` proxies
        without raising `NameError`.

        Args:
            target (Any): Function, method, or class to inspect.
            owner (type[Any] | None): Owning class context for resolving `ForwardRef` instances.

        Returns:
            dict[str, Any]: Mapping of parameter/attribute names to resolved annotations or `ForwardRef`s.
        """
        raw_fn = target.__func__ if inspect.ismethod(target) else target
        fn = raw_fn if isinstance(raw_fn, type) else inspect.unwrap(raw_fn)
        eval_owner: Any = owner
        if eval_owner is None:
            if isinstance(target, type):
                eval_owner = target
            elif inspect.ismethod(target):
                eval_owner = target.__self__ if isinstance(target.__self__, type) else type(target.__self__)
            else:
                eval_owner = fn

        try:
            raw_hints = annotationlib.get_annotations(fn, format=Format.FORWARDREF)
            if any(isinstance(v, str) for v in raw_hints.values()):
                raw_hints = annotationlib.get_annotations(fn, format=Format.FORWARDREF, eval_str=True)
        except Exception:
            return {}

        resolved: dict[str, Any] = {}
        for key, val in raw_hints.items():
            resolved[key] = _eval_forward_ref(val, owner=eval_owner) if isinstance(val, ForwardRef) else val
        return resolved

    @classmethod
    def inspect_callable(
        cls,
        callable_obj: Callable[..., Any],
        *,
        owner: Any = None,
    ) -> tuple[Signature, dict[str, Any]]:
        r"""Return and cache `(Signature, resolved_annotations)` for `callable_obj` in a single `annotationlib` pass."""
        raw_fn = callable_obj.__func__ if inspect.ismethod(callable_obj) else callable_obj
        fn = inspect.unwrap(raw_fn)
        eval_owner: Any = owner
        if eval_owner is None:
            if inspect.ismethod(callable_obj):
                eval_owner = (
                    callable_obj.__self__ if isinstance(callable_obj.__self__, type) else type(callable_obj.__self__)
                )
            else:
                eval_owner = fn
        cache_key = (eval_owner, fn)
        if (cached := cls._SIG_CACHE.get(cache_key)) is not None:
            return cached

        sig = inspect.signature(fn, annotation_format=Format.FORWARDREF)
        if any(isinstance(p.annotation, str) for p in sig.parameters.values()):
            try:
                sig = inspect.signature(fn, annotation_format=Format.FORWARDREF, eval_str=True)
            except Exception:
                pass

        hints: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            ann = param.annotation
            if ann is Parameter.empty:
                continue
            hints[name] = _eval_forward_ref(ann, owner=eval_owner) if isinstance(ann, ForwardRef) else ann

        result = (sig, hints)
        cls._SIG_CACHE[cache_key] = result
        return result

    @classmethod
    def _iter_docstrings(cls, target: Any) -> list[str]:
        r"""Collect cleaned docstrings in priority order from a class MRO, module, or callable."""
        docs: list[str] = []

        if isinstance(target, type):
            classes_to_search = [b for b in target.__mro__ if not cls._is_ignored_base(b)]
        elif isinstance(target, types.ModuleType):
            if mod_doc := inspect.getdoc(target):
                docs.append(mod_doc)
            classes_to_search = []
        elif inspect.ismethod(target):
            owner = target.__self__ if isinstance(target.__self__, type) else type(target.__self__)
            if own_doc := inspect.getdoc(inspect.unwrap(target)):
                if not cls._is_ignored_base(owner) or target.__name__ not in ("__init__", "__call__"):
                    docs.append(own_doc)
            classes_to_search = [b for b in owner.__mro__ if not cls._is_ignored_base(b)]
        elif inspect.isfunction(target):
            if own_doc := inspect.getdoc(inspect.unwrap(target)):
                docs.append(own_doc)
            classes_to_search = []
        else:
            if "__call__" in type(target).__dict__:
                if call_doc := inspect.getdoc(inspect.unwrap(type(target).__dict__["__call__"])):
                    docs.append(call_doc)
            classes_to_search = [b for b in type(target).__mro__ if not cls._is_ignored_base(b)]

        for base in classes_to_search:
            raw_cls_doc = base.__dict__.get("__doc__")
            if isinstance(raw_cls_doc, str) and raw_cls_doc.strip():
                docs.append(inspect.cleandoc(raw_cls_doc))
            for method_name in ("__init__", "__call__"):
                if method_name in base.__dict__:
                    if method_doc := inspect.getdoc(inspect.unwrap(base.__dict__[method_name])):
                        docs.append(method_doc)

        return docs

    @classmethod
    def parse_description(cls, target: Any) -> tuple[str, str]:
        r"""Extract and cache cleaned description and short summary help text from a class or callable.

        Strips parameter and metadata sections (such as `Args:` or `Parameters`) so only
        the prose description remains, without evaluating PEP 649 `__annotate__` closures.

        Args:
            target (type[Any] | Callable[..., Any]): Class or callable to extract documentation from.

        Returns:
            tuple[str, str]: A `(description, help_text)` tuple.
        """
        key = cls._cache_key(target)
        if key is not None and (cached := cls._DESC_CACHE.get(key)) is not None:
            return cached

        result = ("", "")
        for doc in cls._iter_docstrings(target):
            cleaned = cls._SECTION_SPLIT_RE.split(doc, maxsplit=1)[0].strip()
            if cleaned:
                first_line = cleaned.splitlines()[0].strip()
                result = (cleaned, first_line)
                break

        if key is not None:
            cls._DESC_CACHE[key] = result
        return result

    @classmethod
    def parse_parameters(cls, target_cls: type[Any] | Callable[..., Any]) -> dict[str, str]:
        r"""Parse and cache parameter descriptions from a class (and its MRO) or callable.

        Args:
            target_cls (type[Any] | Callable[..., Any]): Target class or callable to inspect.

        Returns:
            dict[str, str]: Mapping of parameter names to their cleaned help strings.
        """
        key = cls._cache_key(target_cls)
        if key is not None and (cached := cls._CACHE.get(key)) is not None:
            return dict(cached)

        docs_to_search = cls._iter_docstrings(target_cls)
        params: dict[str, str] = {}

        for doc in docs_to_search:
            for g_match in cls._GOOGLE_SECTION_RE.finditer(doc):
                section = g_match.group("content")
                for item in cls._GOOGLE_ITEM_RE.finditer(section):
                    name = item.group("name").lstrip("*")
                    first = item.group("first_line").strip()
                    cont_lines = [line.strip() for line in (item.group("cont") or "").splitlines() if line.strip()]
                    desc = " ".join([p for p in (first, *cont_lines) if p]).strip()
                    if desc and name not in params:
                        params[name] = desc

            for n_match in cls._NUMPY_SECTION_RE.finditer(doc):
                section = n_match.group("content")
                for item in cls._NUMPY_ITEM_RE.finditer(section):
                    name = item.group("name").lstrip("*")
                    desc = " ".join(line.strip() for line in item.group("desc").splitlines() if line.strip())
                    if desc and name not in params:
                        params[name] = desc

        if key is not None:
            cls._CACHE[key] = params
        return dict(params)

    @classmethod
    def _unwrap_annotation(
        cls,
        raw_ann: Any,
        *,
        owner: Any = None,
    ) -> tuple[Any, bool, Arg | None]:
        r"""Unwrap `Annotated[T, ...]`, PEP 695 `TypeAliasType`, `ForwardRef`, and `T | None` into `(inner_type, has_none, arg_meta)`."""
        arg_meta: Arg | None = None
        has_none = False
        current = raw_ann

        while True:
            if isinstance(current, ForwardRef):
                evaluated = _eval_forward_ref(current, owner=owner)
                if evaluated is not current and not isinstance(evaluated, ForwardRef):
                    current = evaluated
                    continue
                break

            if isinstance(current, TypeAliasType):
                current = current.__value__
                continue

            origin = get_origin(current)
            if isinstance(origin, TypeAliasType):
                type_params = origin.__type_params__
                type_args = get_args(current)
                param_map = dict(zip(type_params, type_args, strict=False))
                current = _substitute_type_params(origin.__value__, param_map)
                continue

            if origin is Annotated:
                ann_args = get_args(current)
                current = ann_args[0]
                layer_meta: Arg | None = None
                for meta_item in ann_args[1:]:
                    if isinstance(meta_item, Arg):
                        layer_meta = meta_item if layer_meta is None else layer_meta.merge(meta_item)
                    elif isinstance(meta_item, str) and meta_item.strip():
                        str_arg = Arg(help=meta_item.strip())
                        layer_meta = str_arg if layer_meta is None else layer_meta.merge(str_arg)
                if layer_meta is not None:
                    # Outer Annotated layers take precedence over inner TypeAliasType Annotated layers
                    arg_meta = layer_meta if arg_meta is None else layer_meta.merge(arg_meta)
                continue

            if origin in (types.UnionType, Union):
                union_args = [
                    _eval_forward_ref(a, owner=owner) if isinstance(a, ForwardRef) else a for a in get_args(current)
                ]
                non_none = [a for a in union_args if a is not type(None)]
                if len(non_none) < len(union_args):
                    has_none = True
                if len(non_none) == 1:
                    current = non_none[0]
                    continue
                if union_args:
                    acc = non_none[0] if non_none else type(None)
                    for nxt in non_none[1:]:
                        acc = acc | nxt
                    current = acc
                break

            break

        # Evaluate ForwardRef element args inside generic containers when resolvable
        container_origin = get_origin(current)
        if container_origin in (list, tuple, set, Sequence, Iterable, AbstractSet):
            c_args = get_args(current)
            if c_args and any(isinstance(a, ForwardRef) for a in c_args):
                eval_args = tuple(_eval_forward_ref(a, owner=owner) if isinstance(a, ForwardRef) else a for a in c_args)
                try:
                    current = container_origin[eval_args[0] if len(eval_args) == 1 else eval_args]
                except TypeError:
                    pass

        return current, has_none, arg_meta

    @classmethod
    def _get_dataclass_specs(cls, dc_cls: type[Any]) -> list[ParameterSpec]:
        r"""Introspect fields of a `@dataclass` type into `ParameterSpec` items via `annotationlib`."""
        type_hints = cls.resolve_annotations(dc_cls, owner=dc_cls)
        for base in dc_cls.__mro__[1:]:
            if dataclasses.is_dataclass(base):
                base_hints = cls.resolve_annotations(base, owner=dc_cls)
                for k, v in base_hints.items():
                    type_hints.setdefault(k, v)

        param_docs = cls.parse_parameters(dc_cls)
        specs: list[ParameterSpec] = []

        for f in dataclasses.fields(dc_cls):
            if not f.init:
                continue

            raw_ann = type_hints.get(f.name, f.type)
            annotation, has_none, arg_meta = cls._unwrap_annotation(raw_ann, owner=dc_cls)

            if f.default is not dataclasses.MISSING:
                default: Any = f.default
                has_default = True
            elif f.default_factory is not dataclasses.MISSING:
                default = f.default_factory()
                has_default = True
            else:
                default = inspect.Parameter.empty
                has_default = False

            kind = inspect.Parameter.KEYWORD_ONLY if f.kw_only else inspect.Parameter.POSITIONAL_OR_KEYWORD
            field_help = str(f.metadata["help"]) if "help" in f.metadata else ""
            meta_help = arg_meta.help if arg_meta is not None else ""
            doc = field_help or meta_help or param_docs.get(f.name, "")

            specs.append(
                ParameterSpec(
                    name=f.name,
                    annotation=annotation,
                    default=default,
                    docstring=doc,
                    is_optional=has_none or has_default,
                    kind=kind,
                    arg_meta=arg_meta,
                )
            )

        return specs

    @classmethod
    def _resolve_callable(cls, target: type[Any] | Callable[..., Any]) -> Callable[..., Any]:
        r"""Resolve the underlying callable to inspect for signature parameters."""
        if isinstance(target, type):
            is_cmd_subclass = any(
                b.__name__ in ("BaseCommand", "Command") and b.__module__.startswith("clio") for b in target.__mro__
            )
            if is_cmd_subclass:
                return target.__call__
            has_custom_init = any("__init__" in b.__dict__ and not cls._is_ignored_base(b) for b in target.__mro__)
            has_custom_call = any("__call__" in b.__dict__ and not cls._is_ignored_base(b) for b in target.__mro__)
            if has_custom_call and not has_custom_init:
                return target.__call__
            return target.__init__
        return target

    @classmethod
    def get_parameter_specs(cls, target: type[Any] | Callable[..., Any]) -> list[ParameterSpec]:
        r"""Introspect and cache signature, `annotationlib` annotations, defaults, and docstrings into `ParameterSpec` items.

        Supports classes, `@dataclass` types (including expanding nested dataclass parameters),
        `Command` subclasses, PEP 695 `type` aliases, `Annotated[T, Arg(...)]` metadata, and standalone callables.

        Args:
            target (type[Any] | Callable[..., Any]): Class or callable to introspect.

        Returns:
            list[ParameterSpec]: Ordered list of introspected parameter specifications.
        """
        key = cls._cache_key(target)
        if key is not None and (cached := cls._SPEC_CACHE.get(key)) is not None:
            return list(cached)

        if isinstance(target, type) and dataclasses.is_dataclass(target):
            is_cmd_subclass = any(
                b.__name__ in ("BaseCommand", "Command") and b.__module__.startswith("clio") for b in target.__mro__
            )
            if not is_cmd_subclass:
                specs = cls._get_dataclass_specs(target)
                if key is not None:
                    cls._SPEC_CACHE[key] = tuple(specs)
                return specs

        callable_obj = cls._resolve_callable(target)
        eval_owner: Any = (
            target
            if isinstance(target, type)
            else (
                callable_obj.__self__
                if inspect.ismethod(callable_obj) and isinstance(callable_obj.__self__, type)
                else (type(callable_obj.__self__) if inspect.ismethod(callable_obj) else inspect.unwrap(callable_obj))
            )
        )
        sig, type_hints = cls.inspect_callable(callable_obj, owner=eval_owner)
        param_docs = cls.parse_parameters(target)
        specs = []

        for idx, param in enumerate(sig.parameters.values()):
            if idx == 0 and param.name in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            raw_ann = type_hints.get(param.name, param.annotation)
            annotation, has_none, arg_meta = cls._unwrap_annotation(raw_ann, owner=eval_owner)

            # Expand nested @dataclass parameters inline into their constituent CLI flags
            if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
                specs.extend(cls._get_dataclass_specs(annotation))
                continue

            has_default = param.default is not inspect.Parameter.empty
            is_optional = has_none or has_default
            doc = (arg_meta.help if arg_meta is not None and arg_meta.help else "") or param_docs.get(param.name, "")

            specs.append(
                ParameterSpec(
                    name=param.name,
                    annotation=annotation,
                    default=param.default,
                    docstring=doc,
                    is_optional=is_optional,
                    kind=param.kind,
                    arg_meta=arg_meta,
                )
            )

        # If target is a @dataclass Command whose __call__(self) takes no extra args, expose its dataclass fields
        if not specs and isinstance(target, type) and dataclasses.is_dataclass(target):
            specs = cls._get_dataclass_specs(target)
        elif (
            not specs
            and inspect.ismethod(target)
            and isinstance(target.__self__, object)
            and dataclasses.is_dataclass(type(target.__self__))
        ):
            specs = cls._get_dataclass_specs(type(target.__self__))

        spec_tuple = tuple(specs)
        if key is not None:
            cls._SPEC_CACHE[key] = spec_tuple
        return specs

    @classmethod
    def inspect(cls, target: type[Any] | Callable[..., Any]) -> DocstringSpec:
        r"""Compile and cache a complete, immutable `DocstringSpec` for a class or callable.

        Args:
            target (type[Any] | Callable[..., Any]): Class or callable to inspect.

        Returns:
            DocstringSpec: Frozen specification with description, help text, parameter docs, and specs.
        """
        key = cls._cache_key(target)
        if key is not None and (cached := cls._DOC_SPEC_CACHE.get(key)) is not None:
            return cached

        description, help_text = cls.parse_description(target)
        params = cls.parse_parameters(target)
        specs = tuple(cls.get_parameter_specs(target))
        merged_params = dict(params)
        for s in specs:
            if s.docstring:
                merged_params[s.name] = s.docstring

        result = DocstringSpec(
            description=description,
            help_text=help_text,
            parameters=types.MappingProxyType(merged_params),
            specs=specs,
        )
        if key is not None:
            cls._DOC_SPEC_CACHE[key] = result
        return result
