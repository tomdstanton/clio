import annotationlib
import argparse
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
from typing import Annotated, Any, ClassVar, Literal, Self, Union, get_args, get_origin

from clio.agent import serialize_for_json


def _parse_bool(value: str) -> bool:
    r"""Parse string CLI argument into boolean value.

    Args:
        value (str): Raw string token from command line.

    Returns:
        bool: Parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If string cannot be interpreted as a boolean.
    """
    normalized = value.strip().lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: '{value}'")


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
        annotation (Any): Resolved type annotation (unwrapped from `Annotated` and optional union).
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
        r"""All CLI flag strings (including short flags from `Arg.short`) for `parser.add_argument`."""
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
    def type_label(self) -> str:
        r"""Human-readable representation of the parameter's annotation via `annotationlib.type_repr`."""
        if self.annotation is inspect.Parameter.empty:
            return "str"
        if isinstance(self.annotation, ForwardRef):
            return self.annotation.__forward_arg__
        return type_repr(self.annotation)

    def _coerce_env_value(self, raw_env: str) -> Any:
        r"""Convert a raw environment variable string according to `self.annotation`."""
        ann = self.annotation
        origin = get_origin(ann)
        if ann is bool:
            return _parse_bool(raw_env)
        if origin in (list, tuple, set, Sequence, Iterable, AbstractSet) or ann in (list, tuple, set):
            parts = [part.strip() for part in raw_env.split(",") if part.strip()]
            item_args = get_args(ann)
            if item_args:
                item_ann = item_args[0]
                if item_ann not in (Any, object) and not isinstance(item_ann, ForwardRef) and callable(item_ann):
                    return [item_ann(p) for p in parts]
            return parts
        if (
            ann not in (inspect.Parameter.empty, Any, object)
            and origin is None
            and not isinstance(ann, ForwardRef)
            and callable(ann)
        ):
            return ann(raw_env)
        return raw_env

    def _build_argparse_kwargs(self) -> dict[str, Any]:
        r"""Construct the keyword dictionary for `ArgumentParser.add_argument`."""
        has_default = self.has_default
        is_pos = self.is_positional
        effective_default = self.default
        meta = self.arg_meta

        if meta is not None and meta.env:
            env_val = os.environ.get(meta.env)
            if env_val is not None:
                effective_default = self._coerce_env_value(env_val)
                has_default = True

        kwargs: dict[str, Any] = {}

        if is_pos:
            if has_default:
                kwargs["nargs"] = "?"
                kwargs["default"] = effective_default
        else:
            if has_default:
                kwargs["default"] = effective_default
            else:
                kwargs["required"] = True

        help_str = self.docstring
        if meta is not None and meta.env:
            env_tag = f"[env: {meta.env}]"
            if env_tag not in help_str:
                help_str = f"{help_str} {env_tag}".strip()
        if help_str:
            kwargs["help"] = help_str

        if meta is not None and meta.metavar is not None:
            kwargs["metavar"] = meta.metavar

        ann = self.annotation
        origin = get_origin(ann)

        if ann is bool:
            if is_pos:
                kwargs["type"] = _parse_bool
            elif effective_default is False:
                kwargs["action"] = "store_true"
                kwargs.pop("metavar", None)
            else:
                kwargs["action"] = argparse.BooleanOptionalAction
                kwargs.pop("metavar", None)
        elif origin is Literal:
            choices = list(get_args(ann))
            kwargs["choices"] = choices
            if choices:
                choice_types = {type(c) for c in choices}
                if len(choice_types) == 1:
                    kwargs["type"] = next(iter(choice_types))
        elif origin in (list, tuple, set, Sequence, Iterable, AbstractSet) or ann in (list, tuple, set):
            kwargs["nargs"] = "+" if (is_pos or not has_default) else "*"
            item_args = get_args(ann)
            if item_args:
                item_ann = item_args[0]
                if get_origin(item_ann) is Literal:
                    choices = list(get_args(item_ann))
                    kwargs["choices"] = choices
                    if choices and len({type(c) for c in choices}) == 1:
                        kwargs["type"] = type(choices[0])
                elif item_ann not in (Any, object) and not isinstance(item_ann, ForwardRef) and callable(item_ann):
                    kwargs["type"] = item_ann
        elif (
            ann not in (inspect.Parameter.empty, Any, object)
            and origin is None
            and not isinstance(ann, ForwardRef)
            and callable(ann)
        ):
            kwargs["type"] = ann

        if meta is not None and meta.choices is not None:
            kwargs["choices"] = list(meta.choices)

        return kwargs

    def to_argparse(self) -> tuple[str, dict[str, Any]]:
        r"""Compile parameter specification into primary `ArgumentParser.add_argument` name and keyword arguments.

        Returns:
            tuple[str, dict[str, Any]]: A `(arg_name, kwargs)` pair ready for `parser.add_argument`.
        """
        return self.flag_name, self._build_argparse_kwargs()

    def to_argparse_args(self) -> tuple[tuple[str, ...], dict[str, Any]]:
        r"""Compile parameter specification into full option flag tuple and keyword arguments.

        Returns:
            tuple[tuple[str, ...], dict[str, Any]]: A `(option_flags, kwargs)` pair supporting short flags.
        """
        flags = self.option_flags
        kwargs = self._build_argparse_kwargs()
        if not self.is_positional and len(flags) > 1:
            kwargs["dest"] = self.name
        return flags, kwargs

    @property
    def is_required(self) -> bool:
        r"""Whether this parameter is required when invoking the command."""
        if self.has_default:
            return False
        if self.arg_meta is not None and self.arg_meta.env and self.arg_meta.env in os.environ:
            return False
        return True

    @classmethod
    def _annotation_to_json_schema(cls, ann: Any) -> dict[str, Any]:
        r"""Map a resolved Python type annotation to a JSON Schema property fragment."""
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

        if origin in (list, tuple, set, Sequence, Iterable, AbstractSet) or ann in (list, tuple, set):
            item_args = get_args(ann)
            items_schema = cls._annotation_to_json_schema(item_args[0]) if item_args else {"type": "string"}
            return {"type": "array", "items": items_schema}

        if origin in (dict, Mapping) or ann in (dict, Mapping):
            return {"type": "object"}

        if ann is bool:
            return {"type": "boolean"}
        if ann is int:
            return {"type": "integer"}
        if ann is float:
            return {"type": "number"}
        if ann is str:
            return {"type": "string"}
        if isinstance(ann, type):
            if issubclass(ann, os.PathLike):
                return {"type": "string", "format": "path"}
            if issubclass(ann, enum.Enum):
                return {"type": "string", "enum": [serialize_for_json(e.value) for e in ann]}

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
        r"""Validate and coerce a raw JSON or Python value into the parameter's annotated type.

        Args:
            value (Any): Raw input value (e.g. from MCP `tools/call` or programmatic invocation).

        Returns:
            Any: Coerced Python value matching `self.annotation`.

        Raises:
            ValueError: If `value` violates allowed choices or cannot be converted.
        """
        if value is None:
            return None

        ann = self.annotation
        origin = get_origin(ann)
        meta = self.arg_meta

        allowed_choices: list[Any] | None = None
        if meta is not None and meta.choices is not None:
            allowed_choices = list(meta.choices)
        elif origin is Literal:
            allowed_choices = list(get_args(ann))

        if ann is bool:
            coerced: Any = value if isinstance(value, bool) else _parse_bool(str(value))
        elif origin in (list, tuple, set, Sequence, Iterable, AbstractSet) or ann in (list, tuple, set):
            raw_items = [p.strip() for p in value.split(",") if p.strip()] if isinstance(value, str) else list(value)
            item_args = get_args(ann)
            item_choices: list[Any] | None = None
            if item_args:
                item_ann = item_args[0]
                if get_origin(item_ann) is Literal:
                    item_choices = list(get_args(item_ann))
                    if item_choices and len({type(c) for c in item_choices}) == 1:
                        item_type = type(item_choices[0])
                        raw_items = [item if isinstance(item, item_type) else item_type(item) for item in raw_items]
                elif item_ann not in (Any, object) and not isinstance(item_ann, ForwardRef) and callable(item_ann):
                    raw_items = [item if isinstance(item, item_ann) else item_ann(item) for item in raw_items]
            if item_choices is not None:
                for item in raw_items:
                    if item not in item_choices:
                        raise ValueError(f"Invalid choice {item!r} for '{self.name}' (choose from {item_choices})")
            if origin is tuple or ann is tuple:
                coerced = tuple(raw_items)
            elif origin in (set, AbstractSet) or ann is set:
                coerced = set(raw_items)
            else:
                coerced = list(raw_items)
        elif origin is Literal:
            if allowed_choices and len({type(c) for c in allowed_choices}) == 1:
                target_type = type(allowed_choices[0])
                coerced = value if isinstance(value, target_type) else target_type(value)
            else:
                coerced = value
        elif (
            ann not in (inspect.Parameter.empty, Any, object)
            and origin is None
            and not isinstance(ann, ForwardRef)
            and callable(ann)
        ):
            coerced = value if isinstance(value, ann) else ann(value)
        else:
            coerced = value

        if allowed_choices is not None and coerced not in allowed_choices:
            raise ValueError(f"Invalid choice {coerced!r} for '{self.name}' (choose from {allowed_choices})")

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

    _parse_bool = staticmethod(_parse_bool)

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
        attempting `ForwardRef.evaluate(owner=...)` for any deferred symbols while preserving unresolved
        `ForwardRef` proxies instead of failing the entire callable.

        Args:
            target (Any): Function, method, or class to inspect.
            owner (type[Any] | None): Owning class context for resolving `ForwardRef` instances.

        Returns:
            dict[str, Any]: Mapping of parameter/attribute names to resolved annotations or `ForwardRef`s.
        """
        fn = target.__func__ if inspect.ismethod(target) else target
        if owner is None:
            if isinstance(target, type):
                owner = target
            elif inspect.ismethod(target):
                owner = target.__self__ if isinstance(target.__self__, type) else type(target.__self__)

        try:
            raw_hints = annotationlib.get_annotations(fn, format=Format.FORWARDREF)
            if any(isinstance(v, str) for v in raw_hints.values()):
                raw_hints = annotationlib.get_annotations(fn, format=Format.FORWARDREF, eval_str=True)
        except Exception:
            return {}

        resolved: dict[str, Any] = {}
        for key, val in raw_hints.items():
            if isinstance(val, ForwardRef):
                try:
                    resolved[key] = val.evaluate(owner=owner)
                except Exception:
                    resolved[key] = val
            else:
                resolved[key] = val
        return resolved

    @classmethod
    def inspect_callable(
        cls,
        callable_obj: Callable[..., Any],
        *,
        owner: type[Any] | None = None,
    ) -> tuple[Signature, dict[str, Any]]:
        r"""Return and cache `(Signature, resolved_annotations)` for `callable_obj` in a single `annotationlib` pass."""
        fn = callable_obj.__func__ if inspect.ismethod(callable_obj) else callable_obj
        if owner is None and inspect.ismethod(callable_obj):
            owner = callable_obj.__self__ if isinstance(callable_obj.__self__, type) else type(callable_obj.__self__)
        cache_key = (owner, fn)
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
            if isinstance(ann, ForwardRef):
                try:
                    hints[name] = ann.evaluate(owner=owner)
                except Exception:
                    hints[name] = ann
            else:
                hints[name] = ann

        result = (sig, hints)
        cls._SIG_CACHE[cache_key] = result
        return result

    @classmethod
    def _iter_docstrings(cls, target: type[Any] | Callable[..., Any]) -> list[str]:
        r"""Collect cleaned docstrings in priority order from a class MRO or callable."""
        docs: list[str] = []

        if isinstance(target, type):
            classes_to_search = [b for b in target.__mro__ if not cls._is_ignored_base(b)]
        elif inspect.ismethod(target):
            owner = target.__self__ if isinstance(target.__self__, type) else type(target.__self__)
            if own_doc := inspect.getdoc(target):
                if not cls._is_ignored_base(owner) or target.__name__ not in ("__init__", "__call__"):
                    docs.append(own_doc)
            classes_to_search = [b for b in owner.__mro__ if not cls._is_ignored_base(b)]
        elif inspect.isfunction(target):
            if own_doc := inspect.getdoc(target):
                docs.append(own_doc)
            classes_to_search = []
        else:
            if "__call__" in type(target).__dict__:
                if call_doc := inspect.getdoc(type(target).__dict__["__call__"]):
                    docs.append(call_doc)
            classes_to_search = [b for b in type(target).__mro__ if not cls._is_ignored_base(b)]

        for base in classes_to_search:
            raw_cls_doc = base.__dict__.get("__doc__")
            if isinstance(raw_cls_doc, str) and raw_cls_doc.strip():
                docs.append(inspect.cleandoc(raw_cls_doc))
            for method_name in ("__init__", "__call__"):
                if method_name in base.__dict__:
                    if method_doc := inspect.getdoc(base.__dict__[method_name]):
                        docs.append(method_doc)

        return docs

    @classmethod
    def parse_description(cls, target: type[Any] | Callable[..., Any]) -> tuple[str, str]:
        r"""Extract and cache cleaned description and short summary help text from a class or callable.

        Strips parameter and metadata sections (such as `Args:` or `Parameters`) so only
        the prose description remains.

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
            # 1. Try Google-style Args:/Parameters:/Attributes: sections
            for g_match in cls._GOOGLE_SECTION_RE.finditer(doc):
                section = g_match.group("content")
                for item in cls._GOOGLE_ITEM_RE.finditer(section):
                    name = item.group("name").lstrip("*")
                    first = item.group("first_line").strip()
                    cont_lines = [line.strip() for line in (item.group("cont") or "").splitlines() if line.strip()]
                    desc = " ".join([p for p in (first, *cont_lines) if p]).strip()
                    if desc and name not in params:
                        params[name] = desc

            # 2. Try NumPy-style Parameters/Attributes sections
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
    def _unwrap_annotation(cls, raw_ann: Any) -> tuple[Any, bool, Arg | None]:
        r"""Unwrap `Annotated[T, ...]` and `T | None` / `Optional[T]` into `(inner_type, has_none, arg_meta)`."""
        arg_meta: Arg | None = None
        has_none = False
        current = raw_ann

        while True:
            origin = get_origin(current)
            if origin is Annotated:
                ann_args = get_args(current)
                current = ann_args[0]
                for meta_item in ann_args[1:]:
                    if isinstance(meta_item, Arg):
                        arg_meta = meta_item if arg_meta is None else arg_meta.merge(meta_item)
                    elif isinstance(meta_item, str) and meta_item.strip():
                        str_arg = Arg(help=meta_item.strip())
                        arg_meta = str_arg if arg_meta is None else arg_meta.merge(str_arg)
                continue

            if origin in (types.UnionType, Union):
                union_args = get_args(current)
                non_none = [a for a in union_args if a is not type(None)]
                if len(non_none) < len(union_args):
                    has_none = True
                if len(non_none) == 1:
                    current = non_none[0]
                    continue
            break

        return current, has_none, arg_meta

    @staticmethod
    def _unwrap_optional(raw_ann: Any) -> tuple[Any, bool]:
        r"""Unwrap `T | None` / `Optional[T]` (and `Annotated`) into `(inner_type, has_none)`."""
        inner, has_none, _ = DocstringParser._unwrap_annotation(raw_ann)
        return inner, has_none

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
            annotation, has_none, arg_meta = cls._unwrap_annotation(raw_ann)

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
        `Command` subclasses, `Annotated[T, Arg(...)]` metadata, and standalone callables.

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
        owner = target if isinstance(target, type) else None
        sig, type_hints = cls.inspect_callable(callable_obj, owner=owner)
        param_docs = cls.parse_parameters(target)
        specs = []

        for idx, param in enumerate(sig.parameters.values()):
            if idx == 0 and param.name in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            raw_ann = type_hints.get(param.name, param.annotation)
            if (
                raw_ann is argparse.Namespace
                or raw_ann in ("argparse.Namespace", "Namespace")
                or (isinstance(raw_ann, ForwardRef) and raw_ann.__forward_arg__ in ("argparse.Namespace", "Namespace"))
            ):
                continue

            annotation, has_none, arg_meta = cls._unwrap_annotation(raw_ann)

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

    @classmethod
    def backfill_help(
        cls,
        parser: argparse.ArgumentParser,
        target: type[Any] | Callable[..., Any] | Mapping[str, str],
    ) -> None:
        r"""Back-fill missing `help` strings on existing parser actions from parsed docstrings.

        Args:
            parser (argparse.ArgumentParser): Target parser whose actions should be updated.
            target (type[Any] | Callable[..., Any] | Mapping[str, str]): Class, callable, or parameter doc mapping.
        """
        param_docs = target if isinstance(target, Mapping) else cls.parse_parameters(target)
        if not param_docs:
            return
        for action in parser._actions:
            if action.help is None and action.dest in param_docs:
                action.help = param_docs[action.dest]

    @classmethod
    def populate_parser(
        cls,
        parser: argparse.ArgumentParser,
        target: type[Any] | Callable[..., Any] | DocstringSpec | Iterable[ParameterSpec],
    ) -> None:
        r"""Register CLI arguments on an `ArgumentParser` from a class, callable, `DocstringSpec`, or `ParameterSpec` iterable.

        Skips any parameter whose `dest` is already registered on `parser`, and back-fills missing
        `help` text on manually registered arguments from docstrings.

        Args:
            parser (argparse.ArgumentParser): Argument parser to populate.
            target (type[Any] | Callable[..., Any] | DocstringSpec | Iterable[ParameterSpec]): Introspection source.
        """
        if isinstance(target, DocstringSpec):
            specs: Sequence[ParameterSpec] = target.specs
            param_docs: Mapping[str, str] = target.parameters
        elif isinstance(target, type) or callable(target):
            doc_spec = cls.inspect(target)
            specs = doc_spec.specs
            param_docs = doc_spec.parameters
        else:
            specs = tuple(target)
            param_docs = {s.name: s.docstring for s in specs if s.docstring}

        existing_dests = {action.dest for action in parser._actions}

        for spec in specs:
            if spec.name in existing_dests:
                continue
            flags, kwargs = spec.to_argparse_args()
            parser.add_argument(*flags, **kwargs)
            existing_dests.add(spec.name)

        cls.backfill_help(parser, param_docs)
