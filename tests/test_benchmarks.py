import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Annotated, ClassVar

import pytest

from clio import (
    Arg,
    BoxStyle,
    CallPlan,
    Cli,
    CliConfig,
    Colors,
    Command,
    CommandGroup,
    CommandSpec,
    DataclassBinding,
    DataclassCommand,
    DocstringParser,
    DocstringSpec,
    HelpFormatter,
    MarkdownWriter,
    OptionBinding,
    PackageMeta,
    ParameterSpec,
    ScanPlan,
    ScanResult,
    SpinnerConfig,
    Style,
    TargetSpec,
)

DOCS_BENCHMARKS_PATH = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Immutable specification of a single performance micro-benchmark.

    Attributes:
        case_id: Unique short identifier for the benchmark case.
        name: Human-readable title rendered in reports.
        category: Lifecycle phase category (`Compile`, `Build`, `Runtime`, `Docs`).
        description: Concise explanation of what the benchmark measures.
        iterations: Number of timed iterations to execute.
        max_median_us: Maximum allowed median duration in microseconds.
    """

    case_id: str
    name: str
    category: str
    description: str
    iterations: int
    max_median_us: float


@dataclass(frozen=True, slots=True)
class BenchmarkMetric:
    """Immutable statistical summary of a completed benchmark case.

    Attributes:
        case: The benchmark specification that was executed.
        median_us: Median execution latency in microseconds.
        mean_us: Arithmetic mean execution latency in microseconds.
        min_us: Minimum execution latency in microseconds.
        p95_us: 95th-percentile execution latency in microseconds.
        ops_per_sec: Throughput in operations per second (derived from median).
    """

    case: BenchmarkCase
    median_us: float
    mean_us: float
    min_us: float
    p95_us: float
    ops_per_sec: float


@dataclass(frozen=True, slots=True)
class ArchitectureFootprint:
    """Memory and immutability audit entry for a frozen slotted clio dataclass.

    Attributes:
        struct_name: Class name of the specification or config dataclass.
        module_name: Defining module path.
        role: Summary of its role in the stateless/stateful architecture.
        size_bytes: Shallow instance size in bytes (`sys.getsizeof`).
        is_frozen: Whether mutation raises `FrozenInstanceError`.
        has_slots: Whether the instance omits `__dict__` via `__slots__`.
    """

    struct_name: str
    module_name: str
    role: str
    size_bytes: int
    is_frozen: bool
    has_slots: bool


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregated benchmark suite report for test assertions and Zensical rendering.

    Attributes:
        python_version: Python runtime version and build label.
        platform_label: Operating system and CPU architecture description.
        metrics: Tuple of measured benchmark case metrics.
        footprints: Tuple of audited frozen slotted dataclass footprints.
    """

    python_version: str
    platform_label: str
    metrics: tuple[BenchmarkMetric, ...]
    footprints: tuple[ArchitectureFootprint, ...]


@dataclass(frozen=True, slots=True)
class BenchDeployConfig:
    """Deploy a service artifact to the target cluster.

    Args:
        host: Target cluster hostname.
        port: Target cluster port.
        tls: Enable TLS encryption.
    """

    name: ClassVar[str] = "config"

    host: str
    port: Annotated[int, Arg(short="-p", metavar="PORT")] = 8443
    tls: bool = False

    def __call__(self) -> None:
        """Execute the deployment configuration."""


class BenchDeployCommand(Command):
    """Deploy application services to an environment."""

    name = "deploy"

    def __call__(
        self,
        env: str,
        workers: Annotated[int, Arg(short="-w", metavar="N")] = 4,
        *,
        force: Annotated[bool, "Bypass pre-flight checks."] = False,
    ) -> None:
        """Execute the deployment command.

        Args:
            env: Target deployment environment name.
            workers: Number of parallel rollout workers.
        """


class BenchStatusCommand(Command):
    """Check cluster health status."""

    name = "status"

    def __call__(self, *, verbose: bool = False) -> None:
        """Return health check status code.

        Args:
            verbose: Print detailed node diagnostics.
        """


class BenchClusterGroup(CommandGroup):
    """Manage cluster operations."""

    name = "cluster"
    subcommands = [BenchStatusCommand]


def _measure_case(case: BenchmarkCase, fn: Callable[[], object], *, warmup: int = 25) -> BenchmarkMetric:
    """Run warmup and timed iterations for a single benchmark callable."""
    for _ in range(warmup):
        fn()

    samples_us: list[float] = []
    perf_counter = time.perf_counter_ns
    for _ in range(case.iterations):
        start_ns = perf_counter()
        fn()
        elapsed_ns = perf_counter() - start_ns
        samples_us.append(elapsed_ns / 1_000.0)

    samples_us.sort()
    median_us = statistics.median(samples_us)
    mean_us = statistics.fmean(samples_us)
    min_us = samples_us[0]
    p95_idx = min(len(samples_us) - 1, int(len(samples_us) * 0.95))
    p95_us = samples_us[p95_idx]
    ops_per_sec = 1_000_000.0 / median_us if median_us > 0 else 0.0
    return BenchmarkMetric(
        case=case,
        median_us=median_us,
        mean_us=mean_us,
        min_us=min_us,
        p95_us=p95_us,
        ops_per_sec=ops_per_sec,
    )


def collect_architecture_footprints() -> tuple[ArchitectureFootprint, ...]:
    """Audit all core clio frozen slotted dataclasses for size and immutability."""
    arg_meta = Arg(short="-w", help="Number of parallel rollout workers.", metavar="N")
    param_spec = ParameterSpec(
        name="workers",
        annotation=int,
        default=4,
        docstring="Number of parallel rollout workers.",
        is_optional=True,
        arg_meta=arg_meta,
    )
    doc_spec = DocstringSpec(
        description="Deploy application services.",
        help_text="Deploy application services.",
        parameters={"workers": "Number of parallel rollout workers."},
        specs=(param_spec,),
    )
    opt_binding = OptionBinding(spec=param_spec, implicit_bool=None)
    scan_plan = ScanPlan.compile((param_spec,))
    scan_result = ScanResult(values={"workers": 8})
    help_fmt = HelpFormatter()
    dc_binding = DataclassBinding(
        param_name="config",
        dc_cls=BenchDeployConfig,
        field_names=("host", "port", "tls"),
        is_positional_only=False,
    )
    call_plan = CallPlan(
        has_custom_call=True,
        is_async=False,
        pos_params=(),
        kw_params=("env", "workers", "force"),
        dc_bindings=(dc_binding,),
        self_dc_fields=(),
        has_var_kw=False,
        consumed_names=frozenset({"env", "workers", "force"}),
        simple_kw_names=("env", "workers", "force"),
    )
    cmd_spec = CommandSpec(
        name="deploy",
        aliases=(),
        description="Deploy application services.",
        help_text="Deploy application services.",
        docstring_spec=doc_spec,
        call_plan=call_plan,
        scan_plan=scan_plan,
    )
    pkg_meta = PackageMeta(
        name="clio",
        version="0.1.0",
        summary="Lightweight CLI framework.",
        author="Tom Stanton",
        license="MIT",
        homepage="https://github.com/tomdstanton/clio",
        repository="https://github.com/tomdstanton/clio",
        documentation="https://tomdstanton.github.io/clio/",
        urls={},
    )
    cli_cfg = CliConfig(
        prog="clio",
        description="Lightweight CLI framework.",
        epilog=None,
        version="0.1.0",
        formatter=help_fmt,
        package_meta=pkg_meta,
    )
    spinner_cfg = SpinnerConfig()
    target_spec = TargetSpec.parse("path/to/script.py:my_fn")
    style = Style(fg=Colors.BRIGHT_CYAN, bold=True)

    entries: list[tuple[object, str, str]] = [
        (arg_meta, "clio.docstr", "Declarative Annotated[T, Arg(...)] CLI flag & env metadata"),
        (param_spec, "clio.docstr", "Introspected parameter type, default, Arg metadata, and coercion"),
        (doc_spec, "clio.docstr", "Parsed docstring summary, body, and parameter map"),
        (opt_binding, "clio.parser", "Flag-to-ParameterSpec binding entry in ScanPlan"),
        (scan_plan, "clio.parser", "Pre-compiled O(1) strict CLI token scanner"),
        (scan_result, "clio.parser", "Immutable scanned parameter dictionary and mode flags"),
        (help_fmt, "clio.parser", "Zero-dependency ANSI usage and multi-section help renderer"),
        (dc_binding, "clio.command", "Pre-computed @dataclass field binding for CLI hydration"),
        (call_plan, "clio.command", "Pre-compiled sync/async dispatch plan replacing inspect.signature()"),
        (cmd_spec, "clio.command", "Cached command specification (name, help, ScanPlan, CallPlan)"),
        (pkg_meta, "clio.cli", "Extracted PEP 621 package metadata from importlib.metadata"),
        (cli_cfg, "clio.cli", "Immutable root CLI configuration and behavior flags"),
        (spinner_cfg, "clio.spinner", "Visual glyph frames, timing interval, and Style for Spinner"),
        (target_spec, "clio.runner", "Parsed zero-edit file/module target specifier (`script.py:fn`)"),
        (BoxStyle.ROUNDED, "clio.color", "Border glyph specification for boxed help and terminal panels"),
        (style, "clio.color", "Composable ANSI SGR style sequence with pre-compiled prefix"),
    ]

    footprints: list[ArchitectureFootprint] = []
    for instance, mod_name, role in entries:
        cls = type(instance)
        slots: tuple[str, ...] = getattr(cls, "__slots__", ())
        has_slots = bool(slots) and not hasattr(instance, "__dict__")
        first_slot = slots[0]
        is_frozen = False
        try:
            setattr(instance, first_slot, None)
        except FrozenInstanceError:
            is_frozen = True

        footprints.append(
            ArchitectureFootprint(
                struct_name=cls.__name__,
                module_name=mod_name,
                role=role,
                size_bytes=sys.getsizeof(instance),
                is_frozen=is_frozen,
                has_slots=has_slots,
            )
        )
    return tuple(footprints)


@pytest.fixture(scope="session")
def benchmark_report() -> BenchmarkReport:
    """Execute the full clio performance benchmark suite once per pytest session."""
    # 1. Subclass definition (__init_subclass__)
    case_subclass = BenchmarkCase(
        case_id="cmd_subclass_def",
        name="Command Subclass Definition",
        category="Compile",
        description="Define a `Command` subclass (`__init_subclass__` with PEP 649 lazy `__annotate__`).",
        iterations=400,
        max_median_us=250.0,
    )

    def bench_subclass_def() -> type[Command]:
        class DynamicCommand(Command):
            """Dynamic benchmark command."""

            def __call__(self, name: str, count: Annotated[int, Arg(short="-c")] = 1) -> None:
                """Run dynamic command."""

        return DynamicCommand

    # 2. Cold CommandSpec.compile + DocstringParser.inspect + ScanPlan.compile via annotationlib
    case_cold_spec = BenchmarkCase(
        case_id="spec_compile_cold",
        name="Cold CommandSpec Compilation",
        category="Compile",
        description="Single-pass cold introspection (`annotationlib` + `ScanPlan.compile` + `CallPlan.compile`).",
        iterations=250,
        max_median_us=2000.0,
    )

    def bench_cold_spec() -> CommandSpec:
        DocstringParser.clear_cache()
        BenchDeployCommand._spec = None
        return BenchDeployCommand.get_spec()

    BenchDeployCommand.get_spec()

    # 3. Warm Command() instantiation
    case_warm_init = BenchmarkCase(
        case_id="cmd_init_warm",
        name="Warm Command Instantiation",
        category="Runtime",
        description="Instantiate a `Command` subclass when `CommandSpec` is already cached on the class.",
        iterations=2000,
        max_median_us=50.0,
    )

    def bench_warm_init() -> BenchDeployCommand:
        return BenchDeployCommand()

    # 4. DataclassCommand compilation
    case_dc_compile = BenchmarkCase(
        case_id="dataclass_binding_compile",
        name="DataclassCommand Spec Compilation",
        category="Compile",
        description="Wrap a `@dataclass(frozen=True, slots=True)` via `Command.from_dataclass()` with cached `DocstringSpec`.",
        iterations=300,
        max_median_us=2000.0,
    )

    def bench_dc_compile() -> DataclassCommand:
        return Command.from_dataclass(BenchDeployConfig, name="config")

    # 5. Cli + lazy subcommand tree registration
    case_cli_build = BenchmarkCase(
        case_id="cli_build_tree",
        name="Lazy CLI Route Indexing",
        category="Build",
        description="Construct `Cli` and lazily index `Command`, `CommandGroup`, and `@dataclass` routes without evaluating `__annotate__`.",
        iterations=500,
        max_median_us=250.0,
    )

    def bench_cli_build() -> Cli:
        cli = Cli(prog="bench", description="Benchmark CLI", version="1.0.0")
        cli.add_command(BenchDeployCommand)
        cli.add_command(BenchClusterGroup)
        cli.add_command(BenchDeployConfig)
        return cli

    runtime_cli = bench_cli_build()

    # 6. End-to-end Cli.run() command dispatch
    case_cli_dispatch = BenchmarkCase(
        case_id="cli_run_dispatch",
        name="ScanPlan Parse & CallPlan Dispatch",
        category="Runtime",
        description="End-to-end `cli.run([...])` native `ScanPlan.parse()` and fast-path `CallPlan.invoke()` dispatch.",
        iterations=1000,
        max_median_us=250.0,
    )

    def bench_cli_dispatch() -> None:
        runtime_cli.run(["deploy", "production", "-w", "8", "--force"])

    # 7. End-to-end Cli.run() frozen slotted @dataclass hydration
    case_cli_dataclass = BenchmarkCase(
        case_id="cli_run_dataclass",
        name="Frozen @dataclass CLI Hydration",
        category="Runtime",
        description="End-to-end `cli.run([...])` hydrating a `@dataclass(frozen=True, slots=True)` payload.",
        iterations=1000,
        max_median_us=250.0,
    )

    def bench_cli_dataclass() -> None:
        runtime_cli.run(["config", "api.internal", "-p", "9443", "--tls"])

    # 8. Color Style composition & ANSI measurement
    case_color_style = BenchmarkCase(
        case_id="color_style_render",
        name="Style Composition & ANSI Helpers",
        category="Runtime",
        description="Compose frozen `Style` objects, format text, and compute `Colors.visible_len()`.",
        iterations=1500,
        max_median_us=100.0,
    )
    base_style = Style(fg=Colors.BRIGHT_CYAN) | Style(bold=True)

    def bench_color_style() -> int:
        styled = base_style("--workers")
        return Colors.visible_len(styled)

    # 9. Zero-dependency Markdown generation
    case_to_markdown = BenchmarkCase(
        case_id="cli_to_markdown",
        name="Zero-Dependency Markdown Generation",
        category="Docs",
        description="Render complete CLI reference documentation via `cli.to_markdown()`.",
        iterations=300,
        max_median_us=2500.0,
    )

    def bench_to_markdown() -> str:
        return runtime_cli.to_markdown()

    # 10. Boxed multi-column ANSI --help rendering
    case_boxed_help = BenchmarkCase(
        case_id="cli_format_help_boxed",
        name="Boxed Multi-Column ANSI Help",
        category="Docs",
        description="Render Unicode boxed multi-column `--help` panels via `HelpFormatter` and `Colors.box()`.",
        iterations=400,
        max_median_us=1500.0,
    )
    deploy_leaf = BenchDeployCommand()
    bench_fmt = HelpFormatter(boxed=True)

    def bench_boxed_help() -> str:
        return bench_fmt.format_help(
            "bench deploy",
            description=deploy_leaf.description,
            scan_plan=deploy_leaf.get_scan_plan(),
            agent_mode=False,
        )

    metrics = (
        _measure_case(case_subclass, bench_subclass_def),
        _measure_case(case_cold_spec, bench_cold_spec),
        _measure_case(case_warm_init, bench_warm_init),
        _measure_case(case_dc_compile, bench_dc_compile),
        _measure_case(case_cli_build, bench_cli_build),
        _measure_case(case_cli_dispatch, bench_cli_dispatch),
        _measure_case(case_cli_dataclass, bench_cli_dataclass),
        _measure_case(case_color_style, bench_color_style),
        _measure_case(case_to_markdown, bench_to_markdown),
        _measure_case(case_boxed_help, bench_boxed_help),
    )

    py_ver = f"Python {platform.python_version()} ({platform.python_implementation()})"
    plat_label = f"{platform.system()} {platform.release()} ({platform.machine()})"

    return BenchmarkReport(
        python_version=py_ver,
        platform_label=plat_label,
        metrics=metrics,
        footprints=collect_architecture_footprints(),
    )


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    """Render the BenchmarkReport into a Zensical-compatible Markdown page."""
    chart_labels = ", ".join(f'"{m.case.case_id}"' for m in report.metrics)
    chart_values = ", ".join(f"{m.median_us:.2f}" for m in report.metrics)
    max_y = max(max(m.median_us for m in report.metrics) * 1.15, 50.0)

    metric_rows = [
        (
            f"**{m.case.name}**<br><small>{m.case.description}</small>",
            f"`{m.case.category}`",
            f"`{m.median_us:.2f}`",
            f"`{m.mean_us:.2f}`",
            f"`{m.p95_us:.2f}`",
            f"`{m.min_us:.2f}`",
            f"`{m.ops_per_sec:,.0f}`",
            f"`<= {m.case.max_median_us:.0f}`",
        )
        for m in report.metrics
    ]
    footprint_rows = [
        (
            f"[`{fp.struct_name}`](reference.md)",
            f"`{fp.module_name}`",
            f"`{fp.size_bytes}`",
            "`Yes`" if fp.is_frozen else "`No`",
            "`Yes`" if fp.has_slots else "`No`",
            fp.role,
        )
        for fp in report.footprints
    ]
    comparison_rows = (
        (
            "**Runtime Dependencies**",
            "**0** (`stdlib` + `annotationlib` only)",
            "`typing-extensions`",
            "`attrs`, `rich`, `docstring-parser`",
            "`click`, `rich`, `shellingham`, `typing-extensions`",
        ),
        (
            "**Annotation Engine**",
            "**Native PEP 649 `__annotate__` + `Format.FORWARDREF` + PEP 695 `type`** (lazy on matched route only)",
            "Pydantic-style class field evaluation across tree",
            "`typing.get_type_hints` + `Annotated` inspection",
            "`typing.get_type_hints` + Click `OptionInfo`",
        ),
        (
            "**Async & Spinners**",
            "**Pay-only-when-used `async` + thread-backed `Spinner`** (`CLIO_AGENT=1` / `--json` aware)",
            "Always-async `asyncio.run()` + `asyncio` spinner",
            "Sync-first (`rich` spinners external)",
            "Sync-only unless manually wrapped",
        ),
        (
            "**Dispatch Overhead**",
            "**Native `ScanPlan` + `CallPlan`** — zero `argparse` / `Namespace` / runtime `inspect`",
            "Class hydration + coroutine loop startup",
            "Runtime token coercion & binding",
            "Click context stack + callback wrapper frames",
        ),
        (
            "**Help & Error Rendering**",
            "Direct ANSI SGR (`HelpFormatter`), lazy `difflib` on error only",
            "Custom ANSI formatter",
            "`rich` console layout & table rendering",
            "`rich` traceback & box rendering",
        ),
        (
            "**Docs & Agent Native**",
            "Zero-edit `python -m clio script.py[:fn]`, `to_markdown()`, `to_json_schema()`, `SKILL.md`, stdio `McpServer`",
            "Docs only",
            "Sphinx / MkDocs plugin",
            "`typer-cli` external package",
        ),
    )

    md = (
        MarkdownWriter()
        .heading("Performance & Architecture Benchmarks", level=1)
        .paragraph(
            "`clio` is engineered around a strict **Stateless Specification / Stateful Execution** split\n"
            "built natively on Python 3.14's `annotationlib` (PEP 649 / PEP 749) and a custom zero-`argparse`\n"
            "token scanner (`ScanPlan`). Expensive introspection (`annotationlib`, `inspect.signature`,\n"
            "PEP 695 `TypeAliasType` unwrapping, docstring parsing, `Annotated[T, Arg(...)]` extraction,\n"
            "and `@dataclass` field binding) happens **lazily on the matched command path only** and is\n"
            "cached in immutable `@dataclass(frozen=True, slots=True)` specifications."
        )
        .admonition(
            "info",
            "Benchmark Environment",
            (
                f"- **Runtime**: `{report.python_version}`",
                f"- **Platform**: `{report.platform_label}`",
                "- **Suite**: `tests/test_benchmarks.py` (enforced in `pytest` and `prek.toml`)",
            ),
        )
        .heading("Lifecycle Architecture", level=2)
        .code_block(
            "flowchart LR\n"
            '    A["Lazy Route Indexing\\n(Cli.add_command)"] --> B["O(depth) Subcommand Match\\n(Zero Sibling __annotate__)"]\n'
            '    B --> C["Single-Pass Spec Compile\\n(annotationlib + ScanPlan + CallPlan)"]\n'
            '    C --> D["O(1) Token Scan & Dispatch\\n(ScanPlan.parse + CallPlan.invoke)"]',
            lang="mermaid",
        )
        .heading("Micro-Benchmark Results", level=2)
        .table(
            (
                "Benchmark",
                "Category",
                "Median (µs)",
                "Mean (µs)",
                "P95 (µs)",
                "Min (µs)",
                "Throughput (ops/s)",
                "Budget (µs)",
            ),
            metric_rows,
            alignments=("left", "left", "right", "right", "right", "right", "right", "right"),
        )
        .heading("Median Latency Breakdown (µs)", level=3)
        .code_block(
            "xychart-beta\n"
            '    title "clio Lifecycle Median Latency (microseconds, lower is better)"\n'
            f"    x-axis [{chart_labels}]\n"
            f'    y-axis "Latency (us)" 0 --> {max_y:.0f}\n'
            f"    bar [{chart_values}]",
            lang="mermaid",
        )
        .heading("Frozen Slotted Specification Footprint", level=2)
        .paragraph(
            "Every specification and configuration object in `clio` uses `@dataclass(frozen=True, slots=True)`\n"
            "to eliminate per-instance `__dict__` allocation and prevent accidental state mutation across runs."
        )
        .table(
            ("Dataclass", "Module", "Shallow Size (Bytes)", "Frozen", "Slotted", "Architectural Role"),
            footprint_rows,
            alignments=("left", "left", "right", "center", "center", "left"),
        )
        .heading("Why `clio` Stays Fast vs. `clypi`, `cyclopts` & `typer`", level=2)
        .table(("Dimension", "`clio` (Python 3.14+)", "`clypi`", "`cyclopts`", "`typer`"), comparison_rows)
        .heading("Running & Updating Benchmarks", level=2)
        .paragraph("Benchmarks execute automatically during `pytest` and `prek` to enforce latency budgets:")
        .code_block(
            "# Run tests and verify performance budgets\n"
            "uv run pytest\n\n"
            "# Regenerate docs/benchmarks.md with fresh measurements\n"
            "uv run pytest --update-benchmarks",
            lang="bash",
        )
    )
    return f"{md.render()}\n"


def test_all_specs_are_frozen_and_slotted(benchmark_report: BenchmarkReport) -> None:
    """Verify every core clio specification dataclass is both frozen and slotted."""
    assert len(benchmark_report.footprints) == 16
    for footprint in benchmark_report.footprints:
        assert footprint.is_frozen, f"{footprint.struct_name} must be frozen=True"
        assert footprint.has_slots, f"{footprint.struct_name} must be slots=True (no __dict__)"
        assert footprint.size_bytes <= 160, f"{footprint.struct_name} shallow size unexpectedly large"


def test_performance_budgets(benchmark_report: BenchmarkReport) -> None:
    """Assert that every clio lifecycle benchmark meets its latency budget."""
    for metric in benchmark_report.metrics:
        assert metric.median_us <= metric.case.max_median_us, (
            f"Benchmark '{metric.case.case_id}' exceeded budget: "
            f"{metric.median_us:.2f} us > {metric.case.max_median_us:.2f} us"
        )


def test_benchmark_docs_rendered(benchmark_report: BenchmarkReport, update_benchmarks: bool) -> None:
    """Ensure docs/benchmarks.md exists, is updated when requested, and contains all benchmark cases."""
    rendered = render_benchmark_markdown(benchmark_report)
    if update_benchmarks or not DOCS_BENCHMARKS_PATH.exists():
        DOCS_BENCHMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOCS_BENCHMARKS_PATH.write_text(rendered, encoding="utf-8")

    content = DOCS_BENCHMARKS_PATH.read_text(encoding="utf-8")
    for metric in benchmark_report.metrics:
        assert metric.case.name in content
        assert metric.case.case_id in content
    for footprint in benchmark_report.footprints:
        assert f"`{footprint.struct_name}`" in content
