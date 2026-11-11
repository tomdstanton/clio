# clio

**Lightweight, zero-dependency, agent-native CLI framework for Python 3.14+.**

`clio` replaces `argparse` with a native **Stateless Specification / Stateful Execution** engine powered by `@dataclass(frozen=True, slots=True)` specifications (`ScanPlan`, `CallPlan`, `CommandSpec`, `HelpFormatter`) and Python 3.14's `annotationlib`.

---

## Key Features

- **Zero Runtime Dependencies**: Pure Python 3.14+ stdlib (`annotationlib`, `dataclasses`, `inspect`, `importlib.metadata`, `threading`).
- **End-to-End PEP 649 & PEP 695 Introspection**: Preserves deferred `__annotate__` closures across command registration and `--help` (evaluating annotations *only* on the matched subcommand route), resolves nested `ForwardRef`s via `Format.FORWARDREF`, and supports reusable PEP 695 `type` aliases (`typing.TypeAliasType`).
- **Native `ScanPlan` Token Scanner & `HelpFormatter`**: Strict O(1) `--flag` / `-f` / `--flag=val` scanning, `--no-<flag>` and `--<flag>=true|false` boolean overrides, negative numeric literals (`-42`, `-1.5`), `--` end-of-options sentinel, variadic collections, fixed `tuple[T1, T2]` arity, and fuzzy `difflib` error suggestions.
- **Pay-Only-When-Used `async` Execution**: Synchronous commands execute directly with zero `asyncio` import or event-loop overhead; `async def __call__` commands run lazily via `asyncio.run()` in `Cli.run()` or natively on an active loop via `await Cli.run_async()` / `await Command.run_async()`.
- **Zero-Dependency, Agent-Aware `Spinner`**: Thread-backed braille terminal [`Spinner`](reference.md#clio.spinner.Spinner), [`@spinner`](reference.md#clio.spinner.spinner) decorator, and [`Cli.spinner()`](reference.md#clio.cli.Cli.spinner) supporting both `with` and `async with`, automatic TTY cursor restoration, static `CLIO_AGENT=1` output, and `--json` message capture.
- **Strict ABC, `@dataclass`, & Plain-Function Commands**: Explicit [`BaseCommand`](reference.md#clio.command.BaseCommand), [`Command`](reference.md#clio.command.Command), [`CommandGroup`](reference.md#clio.command.CommandGroup), [`DataclassCommand`](reference.md#clio.command.DataclassCommand), and [`FunctionCommand`](reference.md#clio.command.FunctionCommand) contracts, plus single-command [`Command.run()`](reference.md#clio.command.Command.run), `Cli(command=...)`, and `Cli(commands=[my_fn, ...])`.
- **Zero-Edit Target Runner (`python -m clio` / `clio`)**: Wrap any existing Python function, `@dataclass`, or whole module in any repository (`uv run -m clio path/to/script.py:my_fn --dry-run --json` or `uv run -m clio path/to/script.py --mcp`) without editing the target script.
- **Agent-Native Schema, Tool, Skill & MCP Support**: Export JSON Schemas ([`to_json_schema()`](reference.md#clio.cli.Cli.to_json_schema)), LLM/MCP tool definitions ([`to_tools()`](reference.md#clio.cli.Cli.to_tools)), Agent [`SKILL.md`](reference.md#clio.cli.Cli.to_skill_md) files, `--json` / `--dry-run` envelopes ([`AgentResult`](reference.md#clio.agent.AgentResult)), and a zero-dependency stdio MCP server ([`McpServer`](reference.md#clio.agent.McpServer)).

---

## Quick Start

### 1. Subclassing `Command` with PEP 695 `type` Aliases & `Spinner`

```python
import asyncio
from typing import Annotated
from clio import Arg, Cli, Command

type WorkerCount = Annotated[int, Arg(short="-w", metavar="N", env="WORKERS")]


class FetchCommand(Command):
    """Fetch artifacts concurrently from remote endpoints."""

    name = "fetch"

    async def __call__(
        self,
        urls: list[str],
        *,
        workers: WorkerCount = 4,
    ) -> list[str]:
        """Run concurrent fetch tasks.

        Args:
            urls: Target URLs to fetch.
            workers: Maximum parallel worker count.
        """
        assert self.cli is not None
        async with self.cli.spinner(f"Fetching {len(urls)} URLs ({workers} workers)") as sp:
            await asyncio.sleep(0.05)
            sp.succeed(f"Fetched {len(urls)} artifacts")
        return urls


cli = Cli(prog="artifactctl", version="0.1.0", commands=[FetchCommand], enable_agent_flags=True)
```

### 2. Zero-Ceremony `@dataclass(frozen=True, slots=True)` & Single-Command Mode

```python
from dataclasses import dataclass
from clio import Command


@dataclass(frozen=True, slots=True)
class ServeConfig:
    """Launch the HTTP server.

    Attributes:
        host: Bind address for incoming connections.
        port: TCP port to listen on.
        reload: Enable live file-watcher reload.
    """

    host: str
    port: int = 8080
    reload: bool = False

    def __call__(self) -> int:
        print(f"Serving on {self.host}:{self.port} (reload={self.reload})")
        return 0


Command.from_dataclass(ServeConfig, name="serve")
```

### 3. Zero-Edit Target Runner (`python -m clio` / `clio`)

Wrap any existing Python function, `@dataclass`, or whole module without importing `clio` or modifying the target script:

```bash
# Validate arguments and emit a structured JSON envelope without side effects
uv run -m clio path/to/script.py:my_fn --dry-run --json --workers 8

# Expose every public function and callable @dataclass in a module as a stdio MCP server
uv run -m clio path/to/script.py --mcp
```

---

## Explore More

- [**Performance Benchmarks**](benchmarks.md) — Micro-benchmark latency tables, charts, and frozen slotted dataclass memory footprints.
- [**API Reference**](reference.md) — Complete `mkdocstrings[python]` API reference for `clio`.
