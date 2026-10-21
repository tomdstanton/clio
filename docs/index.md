# clio

**Lightweight, zero-dependency, object-oriented CLI framework built on `argparse`.**

`clio` pairs Python's standard library `argparse` with a strict **Stateless Specification / Stateful Execution** architecture powered by `@dataclass(frozen=True, slots=True)` specifications.

---

## Key Features

- **Zero Runtime Dependencies**: Pure Python 3.14+ stdlib (`annotationlib`, `argparse`, `dataclasses`, `inspect`, `importlib.metadata`).
- **`annotationlib`-First Introspection (PEP 649 / PEP 749)**: Native lazy `__annotate__` evaluation with `Format.FORWARDREF` resilience, `type_repr` formatting, and declarative [`Annotated[T, Arg(...)]`](reference.md#clio.docstr.Arg) CLI metadata (short flags, metavars, choices, and `os.environ` fallbacks).
- **Frozen Slotted Specifications**: Introspection (`Arg`, `DocstringSpec`, `ParameterSpec`, `CallPlan`, `CommandSpec`, `DataclassBinding`, `CliConfig`, `PackageMeta`, `Style`) happens once in a single pass and is cached in immutable, slotted dataclasses.
- **Strict ABC Command Hierarchy**: Explicit [`BaseCommand`](reference.md#clio.command.BaseCommand), [`Command`](reference.md#clio.command.Command), [`CommandGroup`](reference.md#clio.command.CommandGroup), and [`DataclassCommand`](reference.md#clio.command.DataclassCommand) base classes enforce clean contracts at instantiation time.
- **First-Class `@dataclass` Support**: Register a `@dataclass(frozen=True, slots=True)` directly via `cli.add_command(MyDataclass)` or inject it as a typed parameter into `Command.__call__`.
- **Automatic Docstring-to-CLI Wiring**: [`DocstringParser`](reference.md#clio.docstr.DocstringParser) extracts Google and NumPy parameter descriptions, type annotations, and defaults, automatically populating `argparse` flags and help text.
- **Package Metadata Auto-Discovery**: [`Cli.from_package()`](reference.md#clio.cli.Cli.from_package) reads `pyproject.toml` / `importlib.metadata` for version, summary, author, and URLs.
- **Zero-Dependency Markdown Reference Generator**: Render complete CLI documentation with [`Cli.to_markdown()`](reference.md#clio.cli.Cli.to_markdown).
- **Agent-Native Schema, Tool, Skill & MCP Support**: Export JSON Schemas ([`to_json_schema()`](reference.md#clio.cli.Cli.to_json_schema)), LLM/MCP tool definitions ([`to_tools()`](reference.md#clio.cli.Cli.to_tools)), Agent [`SKILL.md`](reference.md#clio.cli.Cli.to_skill_md) files, structured `--json` / `--dry-run` execution envelopes ([`AgentResult`](reference.md#clio.agent.AgentResult)), token-efficient `CLIO_AGENT=1` error diagnostics, and a zero-dependency stdio MCP server ([`McpServer`](reference.md#clio.agent.McpServer)).

---

## Quick Start

### 1. Subclassing `Command` with `Annotated[T, Arg(...)]`

```python
from typing import Annotated
from clio import Arg, Cli, Command


class GreetCommand(Command):
    """Greet a user from the terminal."""

    def __call__(
        self,
        name: str,
        *,
        loud: Annotated[bool, Arg(short="-l", env="GREET_LOUD")] = False,
    ) -> None:
        """Execute the greeting.

        Args:
            name: Person to greet.
            loud: Shout the greeting in uppercase.
        """
        msg = f"Hello, {name}!"
        print(msg.upper() if loud else msg)


cli = Cli.from_package("clio")
cli.add_command(GreetCommand)
```

### 2. Zero-Ceremony `@dataclass(frozen=True, slots=True)` Commands

```python
from dataclasses import dataclass
from clio import Cli


@dataclass(frozen=True, slots=True)
class ServeConfig:
    """Launch the HTTP server.

    Args:
        host: Bind address for incoming connections.
        port: TCP port to listen on.
        reload: Enable live file-watcher reload.
    """

    host: str
    port: int = 8080
    reload: bool = False

    def run(self) -> int:
        print(f"Serving on {self.host}:{self.port} (reload={self.reload})")
        return 0


cli = Cli(prog_name="app", description="Service runner")
cli.add_command(ServeConfig, name="serve")
```

---

## Explore More

- [**Performance Benchmarks**](benchmarks.md) — Micro-benchmark latency tables, charts, and frozen slotted dataclass memory footprints.
- [**API Reference**](reference.md) — Complete `mkdocstrings[python]` API reference for `clio`.
