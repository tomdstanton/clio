# Performance & Architecture Benchmarks

`clio` is engineered around a strict **Stateless Specification / Stateful Execution** split
built natively on Python 3.14's `annotationlib` (PEP 649 / PEP 749) and a custom zero-`argparse`
token scanner (`ScanPlan`). Expensive introspection (`annotationlib`, `inspect.signature`,
PEP 695 `TypeAliasType` unwrapping, docstring parsing, `Annotated[T, Arg(...)]` extraction,
and `@dataclass` field binding) happens **lazily on the matched command path only** and is
cached in immutable `@dataclass(frozen=True, slots=True)` specifications.

!!! info "Benchmark Environment"
    - **Runtime**: `Python 3.14.3 (CPython)`
    - **Platform**: `Darwin 25.5.0 (arm64)`
    - **Suite**: `tests/test_benchmarks.py` (enforced in `pytest` and `prek.toml`)

## Lifecycle Architecture

```mermaid
flowchart LR
    A["Lazy Route Indexing\n(Cli.add_command)"] --> B["O(depth) Subcommand Match\n(Zero Sibling __annotate__)"]
    B --> C["Single-Pass Spec Compile\n(annotationlib + ScanPlan + CallPlan)"]
    C --> D["O(1) Token Scan & Dispatch\n(ScanPlan.parse + CallPlan.invoke)"]
```

## Micro-Benchmark Results

| Benchmark | Category | Median (µs) | Mean (µs) | P95 (µs) | Min (µs) | Throughput (ops/s) | Budget (µs) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Command Subclass Definition**<br><small>Define a `Command` subclass (`__init_subclass__` with PEP 649 lazy `__annotate__`).</small> | `Compile` | `4.46` | `4.81` | `5.83` | `4.25` | `224,316` | `<= 250` |
| **Cold CommandSpec Compilation**<br><small>Single-pass cold introspection (`annotationlib` + `ScanPlan.compile` + `CallPlan.compile`).</small> | `Compile` | `59.19` | `119.79` | `232.92` | `51.88` | `16,895` | `<= 2000` |
| **Warm Command Instantiation**<br><small>Instantiate a `Command` subclass when `CommandSpec` is already cached on the class.</small> | `Runtime` | `1.67` | `1.88` | `1.75` | `1.58` | `599,880` | `<= 50` |
| **DataclassCommand Spec Compilation**<br><small>Wrap a `@dataclass(frozen=True, slots=True)` via `Command.from_dataclass()` with cached `DocstringSpec`.</small> | `Compile` | `6.77` | `14.36` | `28.71` | `6.29` | `147,700` | `<= 2000` |
| **Lazy CLI Route Indexing**<br><small>Construct `Cli` and lazily index `Command`, `CommandGroup`, and `@dataclass` routes without evaluating `__annotate__`.</small> | `Build` | `12.38` | `13.83` | `13.79` | `5.33` | `80,808` | `<= 250` |
| **ScanPlan Parse & CallPlan Dispatch**<br><small>End-to-end `cli.run([...])` native `ScanPlan.parse()` and fast-path `CallPlan.invoke()` dispatch.</small> | `Runtime` | `12.96` | `19.98` | `39.42` | `10.79` | `77,172` | `<= 250` |
| **Frozen @dataclass CLI Hydration**<br><small>End-to-end `cli.run([...])` hydrating a `@dataclass(frozen=True, slots=True)` payload.</small> | `Runtime` | `11.29` | `13.15` | `16.96` | `10.71` | `88,566` | `<= 250` |
| **Style Composition & ANSI Helpers**<br><small>Compose frozen `Style` objects, format text, and compute `Colors.visible_len()`.</small> | `Runtime` | `0.17` | `0.18` | `0.21` | `0.12` | `5,988,024` | `<= 100` |
| **Zero-Dependency Markdown Generation**<br><small>Render complete CLI reference documentation via `cli.to_markdown()`.</small> | `Docs` | `26.33` | `28.54` | `35.46` | `25.12` | `37,974` | `<= 2500` |

### Median Latency Breakdown (µs)

```mermaid
xychart-beta
    title "clio Lifecycle Median Latency (microseconds, lower is better)"
    x-axis ["cmd_subclass_def", "spec_compile_cold", "cmd_init_warm", "dataclass_binding_compile", "cli_build_tree", "cli_run_dispatch", "cli_run_dataclass", "color_style_render", "cli_to_markdown"]
    y-axis "Latency (us)" 0 --> 68
    bar [4.46, 59.19, 1.67, 6.77, 12.38, 12.96, 11.29, 0.17, 26.33]
```

## Frozen Slotted Specification Footprint

Every specification and configuration object in `clio` uses `@dataclass(frozen=True, slots=True)`
to eliminate per-instance `__dict__` allocation and prevent accidental state mutation across runs.

| Dataclass | Module | Shallow Size (Bytes) | Frozen | Slotted | Architectural Role |
| :--- | :--- | ---: | :---: | :---: | :--- |
| [`Arg`](reference.md) | `clio.docstr` | `80` | `Yes` | `Yes` | Declarative Annotated[T, Arg(...)] CLI flag & env metadata |
| [`ParameterSpec`](reference.md) | `clio.docstr` | `88` | `Yes` | `Yes` | Introspected parameter type, default, Arg metadata, and coercion |
| [`DocstringSpec`](reference.md) | `clio.docstr` | `64` | `Yes` | `Yes` | Parsed docstring summary, body, and parameter map |
| [`OptionBinding`](reference.md) | `clio.parser` | `48` | `Yes` | `Yes` | Flag-to-ParameterSpec binding entry in ScanPlan |
| [`ScanPlan`](reference.md) | `clio.parser` | `64` | `Yes` | `Yes` | Pre-compiled O(1) strict CLI token scanner |
| [`ScanResult`](reference.md) | `clio.parser` | `72` | `Yes` | `Yes` | Immutable scanned parameter dictionary and mode flags |
| [`HelpFormatter`](reference.md) | `clio.parser` | `72` | `Yes` | `Yes` | Zero-dependency ANSI usage and multi-section help renderer |
| [`DataclassBinding`](reference.md) | `clio.command` | `64` | `Yes` | `Yes` | Pre-computed @dataclass field binding for CLI hydration |
| [`CallPlan`](reference.md) | `clio.command` | `104` | `Yes` | `Yes` | Pre-compiled sync/async dispatch plan replacing inspect.signature() |
| [`CommandSpec`](reference.md) | `clio.command` | `88` | `Yes` | `Yes` | Cached command specification (name, help, ScanPlan, CallPlan) |
| [`PackageMeta`](reference.md) | `clio.cli` | `104` | `Yes` | `Yes` | Extracted PEP 621 package metadata from importlib.metadata |
| [`CliConfig`](reference.md) | `clio.cli` | `96` | `Yes` | `Yes` | Immutable root CLI configuration and behavior flags |
| [`SpinnerConfig`](reference.md) | `clio.spinner` | `72` | `Yes` | `Yes` | Visual glyph frames, timing interval, and Style for Spinner |
| [`TargetSpec`](reference.md) | `clio.runner` | `56` | `Yes` | `Yes` | Parsed zero-edit file/module target specifier (`script.py:fn`) |
| [`Style`](reference.md) | `clio.color` | `120` | `Yes` | `Yes` | Composable ANSI SGR style sequence with pre-compiled prefix |

## Why `clio` Stays Fast vs. `clypi`, `cyclopts` & `typer`

| Dimension | `clio` (Python 3.14+) | `clypi` | `cyclopts` | `typer` |
| :--- | :--- | :--- | :--- | :--- |
| **Runtime Dependencies** | **0** (`stdlib` + `annotationlib` only) | `typing-extensions` | `attrs`, `rich`, `docstring-parser` | `click`, `rich`, `shellingham`, `typing-extensions` |
| **Annotation Engine** | **Native PEP 649 `__annotate__` + `Format.FORWARDREF` + PEP 695 `type`** (lazy on matched route only) | Pydantic-style class field evaluation across tree | `typing.get_type_hints` + `Annotated` inspection | `typing.get_type_hints` + Click `OptionInfo` |
| **Async & Spinners** | **Pay-only-when-used `async` + thread-backed `Spinner`** (`CLIO_AGENT=1` / `--json` aware) | Always-async `asyncio.run()` + `asyncio` spinner | Sync-first (`rich` spinners external) | Sync-only unless manually wrapped |
| **Dispatch Overhead** | **Native `ScanPlan` + `CallPlan`** — zero `argparse` / `Namespace` / runtime `inspect` | Class hydration + coroutine loop startup | Runtime token coercion & binding | Click context stack + callback wrapper frames |
| **Help & Error Rendering** | Direct ANSI SGR (`HelpFormatter`), lazy `difflib` on error only | Custom ANSI formatter | `rich` console layout & table rendering | `rich` traceback & box rendering |
| **Docs & Agent Native** | Zero-edit `python -m clio script.py[:fn]`, `to_markdown()`, `to_json_schema()`, `SKILL.md`, stdio `McpServer` | Docs only | Sphinx / MkDocs plugin | `typer-cli` external package |

## Running & Updating Benchmarks

Benchmarks execute automatically during `pytest` and `pre-commit` to enforce latency budgets:

```bash
# Run tests and verify performance budgets
uv run pytest

# Regenerate docs/benchmarks.md with fresh measurements
uv run pytest --update-benchmarks
```
