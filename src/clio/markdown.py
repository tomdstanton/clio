import enum
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Self

from clio.color import Colors
from clio.docstr import ParameterSpec

type TableAlign = Literal["left", "center", "right"]

_ALIGN_SPEC: Mapping[TableAlign, str] = {
    "left": ":---",
    "center": ":---:",
    "right": "---:",
}

_AGENT_GLOBAL_SPECS: tuple[tuple[str, str], ...] = (
    ("--json", "Emit structured JSON execution envelope"),
    ("--dry-run", "Validate arguments and inspect resolved CallPlan without executing"),
    ("--schema", "Print JSON Schema specification and exit"),
    ("--help-md", "Print Markdown reference documentation and exit"),
    ("--skill", "Print Agent SKILL.md specification and exit"),
)


class MarkdownWriter:
    r"""Slotted, zero-dependency GitHub-Flavored Markdown (GFM) and `SKILL.md` document builder."""

    __slots__ = ("_blocks",)

    def __init__(self) -> None:
        r"""Initialize an empty block-buffered `MarkdownWriter`."""
        self._blocks: list[str] = []

    @staticmethod
    def escape_cell(text: str | None) -> str:
        r"""Strip ANSI sequences, collapse newlines, and escape pipe characters for a GFM table cell."""
        if not text:
            return ""
        cleaned = Colors.strip_ansi(text).strip()
        return " ".join(cleaned.splitlines()).replace("|", r"\|")

    @staticmethod
    def spec_type(spec: ParameterSpec) -> str:
        r"""Format the type label or allowed choice literals of `spec` for Markdown tables."""
        choices = spec.choices_tuple
        if choices is not None:
            return ", ".join(f"`{c}`" for c in choices)
        return f"`{spec.type_label}`"

    @staticmethod
    def spec_default(spec: ParameterSpec) -> str:
        r"""Format the default value or `*required*` marker of `spec` for Markdown tables."""
        if spec.is_required:
            return "*required*"
        if not spec.has_default or spec.default is None:
            return "`None`"
        if isinstance(spec.default, enum.Enum):
            return f"`{spec.default.value!r}`"
        return f"`{spec.default!r}`"

    @classmethod
    def spec_description(cls, spec: ParameterSpec) -> str:
        r"""Format a parameter's Markdown description including `[env: ...]` metadata."""
        desc = spec.docstring
        if spec.arg_meta is not None and spec.arg_meta.env:
            env_tag = f"[env: {spec.arg_meta.env}]"
            if env_tag not in desc:
                desc = f"{desc} {env_tag}".strip()
        return cls.escape_cell(desc)

    @classmethod
    def spec_option_row(cls, opt: ParameterSpec) -> tuple[str, str, str, str]:
        r"""Build a 4-column `(Flag, Type, Default, Description)` row for an option `ParameterSpec`."""
        flag_list = list(opt.option_flags)
        if opt.show_negated_in_help and opt.negated_flag is not None and opt.negated_flag not in flag_list:
            flag_list.append(opt.negated_flag)
        flags = ", ".join(f"`{f}`" for f in flag_list)
        return (flags, cls.spec_type(opt), cls.spec_default(opt), cls.spec_description(opt))

    @staticmethod
    def global_option_rows(
        *,
        shadowed: Mapping[str, Any] | None = None,
        version: str | None = None,
        enable_agent_flags: bool = False,
        allow_mcp: bool = False,
    ) -> list[tuple[str, str, str, str]]:
        r"""Build non-shadowed built-in option table rows (`-h`, `-V`, `-v`, and optional agent flags)."""
        skip = shadowed or {}
        rows: list[tuple[str, str, str, str]] = []
        if "-h" not in skip and "--help" not in skip:
            rows.append(("`-h`, `--help`", "`bool`", "`False`", "Show this help message and exit"))
        if "-V" not in skip and "--verbose" not in skip:
            rows.append(("`-V`, `--verbose`", "`bool`", "`False`", "Enable verbose output/progress"))
        if version is not None and "-v" not in skip and "--version" not in skip:
            rows.append(("`-v`, `--version`", "`str`", "`None`", "Show program's version number and exit"))
        if enable_agent_flags:
            for flag, desc in _AGENT_GLOBAL_SPECS:
                if flag not in skip:
                    rows.append((f"`{flag}`", "`bool`", "`False`", desc))
            if allow_mcp and "--mcp" not in skip:
                rows.append(
                    ("`--mcp`", "`bool`", "`False`", "Start stdio Model Context Protocol (MCP) JSON-RPC server")
                )
        return rows

    def frontmatter(self, fields: Mapping[str, str]) -> Self:
        r"""Append a YAML frontmatter block (`--- ... ---`) with single-line sanitized values."""
        if not fields:
            return self
        lines = ["---"]
        for key, val in fields.items():
            clean_val = " ".join(Colors.strip_ansi(val).strip().splitlines())
            lines.append(f"{key}: {clean_val}")
        lines.append("---")
        self._blocks.append("\n".join(lines))
        return self

    def heading(self, text: str, *, level: int = 1) -> Self:
        r"""Append a Markdown ATX heading (`#`..`######`) at `level`."""
        hashes = "#" * max(1, level)
        self._blocks.append(f"{hashes} {Colors.strip_ansi(text).strip()}")
        return self

    def paragraph(self, text: str | None) -> Self:
        r"""Append a stripped prose paragraph if `text` is non-empty."""
        if text:
            cleaned = Colors.strip_ansi(text).strip()
            if cleaned:
                self._blocks.append(cleaned)
        return self

    def code_block(self, code: str, *, lang: str = "text") -> Self:
        r"""Append a fenced code block (` ```lang ... ``` `)."""
        cleaned = Colors.strip_ansi(code).strip()
        self._blocks.append(f"```{lang}\n{cleaned}\n```")
        return self

    def bullets(self, items: Iterable[str]) -> Self:
        r"""Append an unordered `- ` Markdown list block if `items` is non-empty."""
        lines = [f"- {item}" for item in items if item]
        if lines:
            self._blocks.append("\n".join(lines))
        return self

    def admonition(self, kind: str, title: str, lines: Iterable[str]) -> Self:
        r"""Append an indented MkDocs / Zensical admonition block (`!!! kind "title"`)."""
        body = [f"    {line}" for line in lines]
        if body:
            self._blocks.append("\n".join((f'!!! {kind} "{title}"', *body)))
        return self

    def rule(self) -> Self:
        r"""Append a horizontal rule (`---`)."""
        self._blocks.append("---")
        return self

    def raw(self, block: str | None) -> Self:
        r"""Append a pre-rendered Markdown block verbatim if non-empty."""
        if block:
            stripped = block.strip()
            if stripped:
                self._blocks.append(stripped)
        return self

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        *,
        title: str = "",
        alignments: Sequence[TableAlign] | None = None,
        escape_cells: bool = False,
    ) -> Self:
        r"""Append a GFM pipe table (optionally preceded by a bold `title` caption) if `rows` is non-empty."""
        if not rows:
            return self
        aligns = alignments or ("left",) * len(headers)
        header_line = f"| {' | '.join(headers)} |"
        sep_line = f"| {' | '.join(_ALIGN_SPEC[a] for a in aligns)} |"
        lines = [header_line, sep_line]
        for row in rows:
            cells = (self.escape_cell(c) for c in row) if escape_cells else row
            lines.append(f"| {' | '.join(cells)} |")
        table_str = "\n".join(lines)
        self._blocks.append(f"**{title}**\n\n{table_str}" if title else table_str)
        return self

    def positional_table(self, positionals: Sequence[ParameterSpec], *, title: str = "Positional Arguments") -> Self:
        r"""Append a 3-column CLI positional arguments table if `positionals` is non-empty."""
        if not positionals:
            return self
        rows = [(f"`{p.name}`", self.spec_type(p), self.spec_description(p)) for p in positionals]
        return self.table(("Argument", "Type / Choices", "Description"), rows, title=title)

    def option_table(
        self,
        options: Sequence[ParameterSpec],
        *,
        extra_rows: Sequence[Sequence[str]] = (),
        title: str = "Options",
    ) -> Self:
        r"""Append a 4-column CLI options table combining `options` and any `extra_rows`."""
        rows: list[Sequence[str]] = [self.spec_option_row(opt) for opt in options]
        if extra_rows:
            rows.extend(extra_rows)
        return self.table(("Flag", "Type / Choices", "Default", "Description"), rows, title=title)

    def parameter_sections(
        self,
        positionals: Sequence[ParameterSpec],
        options: Sequence[ParameterSpec],
        *,
        extra_option_rows: Sequence[Sequence[str]] = (),
    ) -> Self:
        r"""Append ungrouped positional/option tables followed by any custom `Arg(group=...)` tables."""
        ungrouped_pos = [p for p in positionals if p.group is None]
        ungrouped_opts = [o for o in options if o.group is None]
        self.positional_table(ungrouped_pos).option_table(ungrouped_opts, extra_rows=extra_option_rows)

        grouped: dict[str, list[ParameterSpec]] = {}
        for spec in (*positionals, *options):
            if spec.group is not None:
                grouped.setdefault(spec.group, []).append(spec)

        for group_name, group_specs in grouped.items():
            if all(s.is_positional for s in group_specs):
                self.positional_table(group_specs, title=group_name)
            elif all(not s.is_positional for s in group_specs):
                self.option_table(group_specs, title=group_name)
            else:
                rows = [self.spec_option_row(s) for s in group_specs]
                self.table(("Parameter", "Type / Choices", "Default", "Description"), rows, title=group_name)
        return self

    def command_table(
        self,
        commands: Sequence[tuple[str, Sequence[str], str]],
        *,
        col_label: str = "Command",
        title: str = "Commands",
    ) -> Self:
        r"""Append a 3-column CLI subcommands summary table from `(name, aliases, summary)` tuples."""
        if not commands:
            return self
        rows = [
            (
                f"`{name}`",
                ", ".join(f"`{a}`" for a in aliases) if aliases else "—",
                self.escape_cell(summary),
            )
            for name, aliases, summary in commands
        ]
        return self.table((col_label, "Aliases", "Summary"), rows, title=title)

    def skill_header(
        self,
        prog: str,
        description: str,
        *,
        section_title: str,
        tips: Sequence[str],
    ) -> Self:
        r"""Append standard Agent `SKILL.md` YAML frontmatter, title heading, and execution tips."""
        skill_name = prog.replace(" ", "-").replace("_", "-").lower()
        return (
            self.frontmatter({"name": skill_name, "description": description})
            .heading(f"`{prog}` Agent Skill", level=1)
            .heading(section_title, level=2)
            .bullets(tips)
        )

    def render(self) -> str:
        r"""Join all buffered Markdown blocks with blank-line separators."""
        return "\n\n".join(self._blocks)

    def __str__(self) -> str:
        r"""Return the rendered Markdown string."""
        return self.render()
