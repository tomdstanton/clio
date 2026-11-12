r"""Comprehensive test suite for the clio CLI library."""

import asyncio
import enum
import inspect
import io
import json
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pytest

if TYPE_CHECKING:

    class UnresolvedFutureType:
        """Type defined only during static type checking to test runtime PEP 649 ForwardRef handling."""


from clio import (
    AgentErrorSpec,
    AgentMessage,
    AgentResult,
    Arg,
    BaseCommand,
    BoxStyle,
    CallPlan,
    Cli,
    CliConfig,
    CliError,
    Colors,
    Command,
    CommandGroup,
    CommandSpec,
    DocstringParser,
    DocstringSpec,
    FunctionCommand,
    HelpFormatter,
    MarkdownWriter,
    McpServer,
    OptionBinding,
    PackageMeta,
    ParameterSpec,
    ParseError,
    ScanPlan,
    ScanResult,
    Spinner,
    SpinnerConfig,
    Style,
    TargetSpec,
    ToolSpec,
    build_target_cli,
    run_target,
    run_target_async,
    serialize_for_json,
    spinner,
)
from clio.runner import main as runner_main

# PEP 695 Reusable CLI Type Aliases ------------------------------------------------------------------------------------
type WorkerCount = Annotated[int, Arg(short="-w", metavar="N", env="PEP695_WORKERS")]
type OutputFormat = Literal["json", "tsv", "parquet"]
type TaggedValues[T] = Annotated[list[T], Arg(short="-t", help="Tagged items")]


# Fixtures & Helper Commands -------------------------------------------------------------------------------------------
def _reflow_help(text: str) -> str:
    r"""Collapse ANSI escapes, Unicode box borders, and wrapped whitespace into single-spaced prose."""
    return " ".join(Colors.strip_ansi(text).replace("│", " ").split())


class CustomDomainError(Exception):
    r"""Dummy domain exception for testing `catch_exceptions`."""


@dataclass(frozen=True, slots=True)
class SharedBranchOpts:
    r"""Shared options reusable across subcommands via dataclass injection.

    Attributes:
        shared_flag: Shared branch boolean flag.
    """

    shared_flag: bool = False


class LeafCommand(Command):
    r"""Execute a leaf action.

    Detailed multiline description for the leaf command.

    Args:
        name: Target name.
    """

    aliases = ["lf"]

    def __call__(self, name: Annotated[str, Arg(short="-n")] = "world") -> None:
        if self.cli is not None:
            self.cli.msg(f"Running leaf for {name}")
        print(f"Hello, {name}!")


class ParentBranchCommand(CommandGroup):
    r"""Group of nested subcommands."""

    name = "branch"

    def register_subcommands(self) -> None:
        self.add_subcommand(LeafCommand)


# 1. Colors Tests ------------------------------------------------------------------------------------------------------
def test_colors_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="cannot be instantiated"):
        Colors()


def test_colors_wrap_enabled_and_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Colors, "ENABLED", True)
    styled = Colors.wrap("warning", Colors.BOLD, Colors.BOLD_RED)
    assert styled == f"{Colors.BOLD}{Colors.BOLD_RED}warning{Colors.RESET}"
    assert Colors.wrap("plain") == "plain"
    assert Colors.wrap("", Colors.BOLD) == ""

    monkeypatch.setattr(Colors, "ENABLED", False)
    plain = Colors.wrap("warning", Colors.BOLD_RED)
    assert plain == "warning"


def test_colors_ergonomic_helpers_and_advanced_palettes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Colors, "ENABLED", True)

    assert Colors.bold("hi") == f"{Colors.BOLD}hi{Colors.RESET}"
    assert Colors.dim("hi") == f"{Colors.DIM}hi{Colors.RESET}"
    assert Colors.italic("hi") == f"{Colors.ITALIC}hi{Colors.RESET}"
    assert Colors.underline("hi") == f"{Colors.UNDERLINE}hi{Colors.RESET}"
    assert Colors.strikethrough("hi") == f"{Colors.STRIKETHROUGH}hi{Colors.RESET}"
    assert Colors.red("err") == f"{Colors.RED}err{Colors.RESET}"
    assert Colors.red("err", bold=True) == f"{Colors.BOLD_RED}err{Colors.RESET}"
    assert Colors.green("ok", bold=True) == f"{Colors.BOLD_GREEN}ok{Colors.RESET}"
    assert Colors.yellow("warn") == f"{Colors.YELLOW}warn{Colors.RESET}"
    assert Colors.blue("info") == f"{Colors.BLUE}info{Colors.RESET}"
    assert Colors.magenta("tag") == f"{Colors.MAGENTA}tag{Colors.RESET}"
    assert Colors.cyan("hl", bold=True) == f"{Colors.BOLD_CYAN}hl{Colors.RESET}"
    assert Colors.white("txt") == f"{Colors.WHITE}txt{Colors.RESET}"
    assert Colors.gray("dimmed") == f"{Colors.GRAY}dimmed{Colors.RESET}"

    assert Colors.success("done") == f"{Colors.BOLD_GREEN}done{Colors.RESET}"
    assert Colors.warning("careful") == f"{Colors.BOLD_YELLOW}careful{Colors.RESET}"
    assert Colors.error("failed") == f"{Colors.BOLD_RED}failed{Colors.RESET}"
    assert Colors.info("note") == f"{Colors.BOLD_CYAN}note{Colors.RESET}"
    assert Colors.muted("subtle") == f"{Colors.DIM}subtle{Colors.RESET}"

    composed = Colors.style("alert", fg=Colors.RED, bg=Colors.BG_WHITE, bold=True, underline=True)
    assert composed == f"{Colors.BOLD}{Colors.UNDERLINE}{Colors.RED}{Colors.BG_WHITE}alert{Colors.RESET}"

    assert Colors.fg256(208) == "\033[38;5;208m"
    assert Colors.bg256(235) == "\033[48;5;235m"
    assert Colors.rgb(255, 85, 0) == "\033[38;2;255;85;0m"
    assert Colors.rgb(0, 128, 255, background=True) == "\033[48;2;0;128;255m"
    assert Colors.hex("#ff5500") == "\033[38;2;255;85;0m"
    assert Colors.hex("f50") == "\033[38;2;255;85;0m"

    with pytest.raises(ValueError, match="256-color"):
        Colors.fg256(300)
    with pytest.raises(ValueError, match="RGB"):
        Colors.rgb(256, 0, 0)
    with pytest.raises(ValueError, match="Invalid hex"):
        Colors.hex("#zzzzzz")

    assert Colors.strip_ansi(composed) == "alert"
    assert Colors.visible_len(composed) == 5
    assert Colors.strip_ansi("no ansi") == "no ansi"

    base_style = Style(fg=Colors.CYAN, bold=True)
    accent_style = base_style | Style(underline=True)
    assert accent_style("header") == f"{Colors.BOLD}{Colors.UNDERLINE}{Colors.CYAN}header{Colors.RESET}"
    with pytest.raises(AttributeError):
        base_style.bold = False  # type: ignore


# 2. HelpFormatter & Native Parser Tests -------------------------------------------------------------------------------
def test_help_formatter_renders_clean_synopsis_and_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    monkeypatch.delenv("CLIO_AGENT", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    cli = Cli(
        prog="mytool",
        description="Top level tool",
        epilog="Footer note.",
        version="1.2.3",
        formatter=HelpFormatter(max_width=80, col_width=26),
    )
    cli.add_command(LeafCommand)

    help_output = cli.format_help()

    assert "usage: mytool [options] [subcommand]" in help_output
    assert "{leafcommand" not in help_output
    assert "Top level tool" in help_output
    assert "╭─ 🌎 Global options" in help_output
    assert "╭─ 💬 Commands" in help_output
    assert "leafcommand (lf)" in help_output
    assert "Footer note." in help_output


def test_parse_error_fuzzy_subcommand_suggestion(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    monkeypatch.delenv("CLIO_AGENT", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    cli = Cli(prog="mytool", commands=[LeafCommand])

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["leafcomman"])  # Typo of 'leafcommand'

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage: mytool" in stderr
    assert "Try 'mytool --help' for help." in stderr
    assert "╭─ Error" in stderr
    assert "argument command: invalid choice: 'leafcomman'" in stderr
    assert "💡 Did you mean 'leafcommand'?" in stderr


# 3. Command Hierarchy & Execution Tests -------------------------------------------------------------------------------
def test_command_metadata_auto_population() -> None:
    cmd = LeafCommand()
    assert cmd.name == "leafcommand"
    assert cmd.help_text == "Execute a leaf action."
    assert "Detailed multiline description" in cmd.description


def test_nested_subcommands_and_shared_dataclass_composition() -> None:
    received_args: dict[str, Any] = {}

    class InspectLeaf(Command):
        """Inspect inherited flags."""

        name = "inspect"

        def __call__(self, shared: SharedBranchOpts) -> None:
            assert self.cli is not None
            received_args["verbose"] = self.cli.verbose
            received_args["shared_flag"] = shared.shared_flag

    class CustomBranch(CommandGroup):
        """Custom nested branch."""

        name = "branch"
        subcommands = [InspectLeaf]

    with Cli(prog="mytool") as app:
        app.add_command(CustomBranch)
        app.run(["branch", "inspect", "-V", "--shared-flag"])

    assert received_args["verbose"] is True
    assert received_args["shared_flag"] is True
    assert app.verbose is True


def test_branch_command_without_call_requires_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with Cli(prog="mytool", commands=[ParentBranchCommand]) as app:
        with pytest.raises(SystemExit) as exc_info:
            app.run(["branch"])

    assert exc_info.value.code == 2
    assert "the following arguments are required: branch_subcommand" in capsys.readouterr().err


def test_version_flag_optional_registration(capsys: pytest.CaptureFixture[str]) -> None:
    cli_with_ver = Cli(prog="mytool", version="0.5.0", commands=[LeafCommand])
    with pytest.raises(SystemExit) as exc_info:
        cli_with_ver.run(["--version"])
    assert exc_info.value.code == 0
    assert "mytool 0.5.0" in capsys.readouterr().out

    cli_no_ver = Cli(prog="mytool", commands=[LeafCommand])
    with pytest.raises(SystemExit) as exc_info:
        cli_no_ver.run(["-v"])
    assert exc_info.value.code == 2


# 4. Error Trapping & Context Manager Signal Handling ------------------------------------------------------------------
def test_catch_cli_error_and_custom_exceptions(capsys: pytest.CaptureFixture[str]) -> None:
    class FailingCommand(Command):
        """Command that raises errors."""

        name = "fail"

        def __call__(self, mode: Literal["cli", "domain"] = "cli") -> None:
            if mode == "cli":
                raise CliError("Custom CLI failure", exit_code=42)
            raise CustomDomainError("Database connection dropped")

    cli = Cli(prog="mytool", catch_exceptions=(CustomDomainError,), commands=[FailingCommand])

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["fail", "--mode", "cli"])
    assert exc_info.value.code == 42
    assert "❌ Custom CLI failure" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["fail", "--mode", "domain"])
    assert exc_info.value.code == 1
    assert "❌ Database connection dropped" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("exception_cls", "expected_code", "expected_stderr_snippet"),
    [
        (KeyboardInterrupt, 1, "🛑 Cancelled by user."),
        (PermissionError, 1, "🔒 Permission denied:"),
        (FileNotFoundError, 1, "📄 File not found:"),
    ],
)
def test_context_manager_traps_system_exceptions(
    capsys: pytest.CaptureFixture[str],
    exception_cls: type[BaseException],
    expected_code: int,
    expected_stderr_snippet: str,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        with Cli(prog="mytool"):
            raise exception_cls("test detail")

    assert exc_info.value.code == expected_code
    assert expected_stderr_snippet in capsys.readouterr().err


# 5. File Handle Tracking, Progress & Verbose Logging ------------------------------------------------------------------
def test_open_file_tracking_and_cleanup(tmp_path: Path) -> None:
    test_file = tmp_path / "output.txt"

    with Cli(prog="mytool") as app:
        handle = app.open_file(test_file, mode="w")
        handle.write("data")
        assert not handle.closed
        stdout_handle = app.open_file("-", mode="w")
        stdout_bin_handle = app.open_file("stdout", mode="wb")
        assert stdout_handle is sys.stdout
        assert stdout_bin_handle is sys.stdout.buffer

    assert handle.closed
    assert not sys.stdout.closed
    assert not sys.stdout.buffer.closed
    assert test_file.read_text() == "data"


def test_verbose_msg_and_progress_reporting(capsys: pytest.CaptureFixture[str]) -> None:
    cli = Cli(prog="mytool")

    cli.verbose = False
    cli.msg("Hidden log")
    items = list(cli.progress([1, 2, 3], "Processing"))
    assert items == [1, 2, 3]
    assert capsys.readouterr().err == ""

    cli.verbose = True
    cli.msg("Visible log")
    gen_items = list(cli.progress((x for x in range(2)), "Streaming"))
    assert gen_items == [0, 1]
    stderr = capsys.readouterr().err
    assert "Visible log" in stderr
    assert "Streaming 1/?" in stderr
    assert "Streaming 2/?" in stderr


# 6. DocstringParser & Automatic Argument Introspection Tests ----------------------------------------------------------
def test_docstring_parser_google_and_numpy_mro_extraction() -> None:
    class BaseWorker:
        """Base worker summary.

        Extended base worker documentation.

        Args:
            source (str): Input source path
                with multiline continuation.
            threads (int): Thread pool size.
        """

        def __init__(self, source: str, threads: int = 4) -> None:
            pass

    class DerivedWorker(BaseWorker):
        """Derived worker summary.

        Parameters
        ----------
        threads : int
            Overridden thread count description.
        batch_size : int, optional
            Number of items per batch.

        Returns
        -------
        None
        """

        def __init__(self, source: str, threads: int = 8, batch_size: int = 32) -> None:
            super().__init__(source, threads=threads)

    desc, help_text = DocstringParser.parse_description(DerivedWorker)
    assert help_text == "Derived worker summary."
    assert desc == "Derived worker summary."
    assert "Parameters" not in desc

    params = DocstringParser.parse_parameters(DerivedWorker)
    assert params["source"] == "Input source path with multiline continuation."
    assert params["threads"] == "Overridden thread count description."
    assert params["batch_size"] == "Number of items per batch."

    specs = DocstringParser.get_parameter_specs(DerivedWorker)
    assert [s.name for s in specs] == ["source", "threads", "batch_size"]
    assert specs[0].is_optional is False
    assert specs[0].docstring == "Input source path with multiline continuation."
    assert specs[1].default == 8
    assert specs[1].is_optional is True


def test_command_auto_introspects_call_signature_and_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    captured: dict[str, Any] = {}

    class ProcessCommand(Command):
        """Process dataset files.

        Runs the processing pipeline over target inputs.

        Args:
            input_path (Path): Path to input file.
            mode (str): Execution mode selector.
            tags (list[str]): Tag labels to attach.
            count (int | None): Optional record limit.
            dry_run (bool): Run without writing changes.
            cache (bool): Enable result caching.
            force (bool): Required keyword-only confirmation flag.
        """

        name = "process"

        def __call__(
            self,
            input_path: Path,
            mode: Literal["fast", "accurate"] = "fast",
            tags: list[str] = [],  # noqa: B006
            count: int | None = None,
            dry_run: bool = False,
            cache: bool = True,
            *,
            force: bool,
        ) -> None:
            captured.update(
                input_path=input_path,
                mode=mode,
                tags=tags,
                count=count,
                dry_run=dry_run,
                cache=cache,
                force=force,
            )

    cmd = ProcessCommand()
    assert cmd.help_text == "Process dataset files."
    assert "Runs the processing pipeline" in cmd.description
    assert "Args:" not in cmd.description

    cli = Cli(prog="mytool", commands=[cmd])

    help_text = cmd.format_help(prog_prefix="mytool")
    reflowed_help = _reflow_help(help_text)
    assert "Path to input file." in reflowed_help
    assert "Execution mode selector." in reflowed_help
    assert "--no-cache" in help_text
    assert "--no-dry-run" not in help_text  # Default is False, so --no-dry-run is hidden in help

    cli.run(
        [
            "process",
            "data/sample.csv",
            "--mode",
            "accurate",
            "--tags",
            "alpha",
            "beta",
            "--count",
            "10",
            "--dry-run",
            "--no-cache",
            "--force",
        ]
    )

    assert captured == {
        "input_path": Path("data/sample.csv"),
        "mode": "accurate",
        "tags": ["alpha", "beta"],
        "count": 10,
        "dry_run": True,
        "cache": False,
        "force": True,
    }


def test_annotated_short_flags_and_docstring_help_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    captured: dict[str, Any] = {}

    class HybridCommand(Command):
        """Run hybrid command.

        Args:
            name (str): Documented name help from docstring.
            retries (int): Number of retry attempts.
        """

        name = "hybrid"

        def __call__(
            self,
            name: Annotated[str, Arg(short="-n")] = "default-name",
            retries: int = 3,
        ) -> None:
            captured["name"] = name
            captured["retries"] = retries

    cmd = HybridCommand()
    cli = Cli(prog="mytool", commands=[cmd])

    help_out = cmd.format_help(prog_prefix="mytool")
    reflowed_out = _reflow_help(help_out)
    assert "--name" in help_out and "-n" in help_out
    assert "Documented name help from docstring." in reflowed_out
    assert "--retries" in help_out
    assert "Number of retry attempts." in reflowed_out

    cli.run(["hybrid", "-n", "custom", "--retries", "5"])
    assert captured == {"name": "custom", "retries": 5}


def test_stateless_class_registration_and_frozen_specs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DeclarativeBranch(CommandGroup):
        """Declarative branch using class-level subcommands list."""

        name = "decl"
        subcommands = [LeafCommand]

    spec = LeafCommand.get_spec()
    assert isinstance(spec, CommandSpec)
    assert isinstance(spec.call_plan, CallPlan)
    assert isinstance(spec.scan_plan, ScanPlan)
    assert isinstance(spec.docstring_spec, DocstringSpec)
    assert spec.call_plan.has_custom_call is True
    assert spec.call_plan.is_async is False

    cli = Cli(prog="mytool", commands=[DeclarativeBranch])
    assert isinstance(cli.config, CliConfig)
    cli.run(["decl", "lf", "-n", "stateless"])
    assert "Hello, stateless!" in capsys.readouterr().out


def test_strict_abc_enforcement_on_command_and_base_command() -> None:
    class IncompleteCommand(Command):
        """Command missing __call__ implementation."""

    with pytest.raises(TypeError, match="abstract"):
        IncompleteCommand()  # type: ignore

    with pytest.raises(TypeError, match="abstract"):
        BaseCommand()  # type: ignore

    with pytest.raises(TypeError, match="abstract"):
        Cli(prog="bad", commands=[IncompleteCommand])


def test_dataclass_command_generation_and_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    executed: list[Any] = []

    @dataclass(frozen=True, slots=True)
    class TrainJob:
        """Train a model using a frozen slotted dataclass.

        Attributes:
            dataset (Path): Dataset path on disk.
            epochs (int): Number of training epochs.
            tags (list[str]): Experiment tags.
        """

        name: str = field(default="train", init=False)
        dataset: Path
        epochs: int = 5
        tags: list[str] = field(default_factory=list)

        def __call__(self) -> None:
            executed.append(self)

    @dataclass(frozen=True, slots=True)
    class OptimizerSpec:
        """Optimizer hyperparameters.

        Attributes:
            lr (float): Learning rate.
            weight_decay (float): Weight decay coefficient.
        """

        lr: float = 0.001
        weight_decay: float = 0.01

    class TuneCommand(Command):
        """Tune hyperparameters with injected dataclass config."""

        name = "tune"

        def __call__(self, dataset: Path, opt: OptimizerSpec, *, dry_run: bool = False) -> None:
            executed.append((dataset, opt, dry_run))

    cli = Cli(prog="mltool", commands=[TrainJob, TuneCommand])
    cli.run(["train", "data/train.parquet", "--epochs", "20", "--tags", "v1", "fast"])

    assert len(executed) == 1
    job = executed[0]
    assert isinstance(job, TrainJob)
    assert job.dataset == Path("data/train.parquet")
    assert job.epochs == 20
    assert job.tags == ["v1", "fast"]

    cli.run(["tune", "data/val.parquet", "--lr", "0.05", "--dry-run"])
    assert len(executed) == 2
    ds, opt, dry = executed[1]
    assert ds == Path("data/val.parquet")
    assert opt == OptimizerSpec(lr=0.05, weight_decay=0.01)
    assert dry is True

    handled: list[OptimizerSpec] = []
    opt_cmd = Command.from_dataclass(OptimizerSpec, handler=handled.append, name="opt")
    cli.add_command(opt_cmd)
    cli.run(["opt", "--lr", "0.1", "--weight-decay", "0.0"])
    assert handled == [OptimizerSpec(lr=0.1, weight_decay=0.0)]


def test_zero_dependency_markdown_documentation_generation() -> None:
    class DeployCommand(Command):
        """Deploy service to target environment.

        Args:
            env (str): Target deployment environment.
            replicas (int): Number of service replicas.
        """

        name = "deploy"
        aliases = ["dp"]

        def __call__(self, env: Literal["staging", "prod"], replicas: int = 2) -> None:
            pass

    cli = Cli(
        prog="cloudctl",
        description="Cloud control plane CLI.",
        epilog="Built with clio.",
        version="2.0.0",
        commands=[DeployCommand, ParentBranchCommand],
    )

    md = cli.to_markdown(include_global_options=True)
    assert "# `cloudctl`" in md
    assert "Cloud control plane CLI." in md
    assert "**Global Options**" in md
    assert "`-v`, `--version`" in md
    assert "## `cloudctl deploy`" in md
    assert "*Aliases:* `dp`" in md
    assert "| `env` | `staging`, `prod` | Target deployment environment. |" in md
    assert "| `--replicas` | `int` | `2` | Number of service replicas. |" in md
    assert "## `cloudctl branch`" in md
    assert "### `cloudctl branch leafcommand`" in md
    assert "Built with clio." in md

    standalone_md = DeployCommand().to_markdown()
    assert "## `deploy`" in standalone_md
    assert "`--replicas`" in standalone_md


def test_importlib_metadata_auto_populates_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = Cli.from_package("pytest", commands=[LeafCommand])
    assert app.config.prog == "pytest"
    assert app.config.version is not None
    assert app.config.description is not None
    assert "pytest" in app.config.description.lower()
    assert isinstance(app.config.package_meta, PackageMeta)
    assert app.config.package_meta.name == "pytest"

    with pytest.raises(SystemExit) as exc_info:
        app.run(["--version"])
    assert exc_info.value.code == 0
    assert f"pytest {app.config.version}" in capsys.readouterr().out

    assert PackageMeta.resolve("nonexistent-package-xyz-999") is None
    assert PackageMeta.resolve("") is None


# 7. Python 3.14 annotationlib, PEP 649 Lazy __annotate__ & PEP 695 Type Aliases --------------------------------------
def test_pep649_lazy_annotate_preserved_across_registration_help_and_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    annotate_evals = 0

    def _track_annotate_eval() -> type[int]:
        nonlocal annotate_evals
        annotate_evals += 1
        return int

    class UntouchedTrackedCmd(Command):
        """Command whose PEP 649 __annotate__ tracks evaluation count."""

        name = "tracked"
        aliases = ["tr"]

        def __call__(self, value: _track_annotate_eval()) -> int:  # type: ignore
            return value + 1

    ran_safe = False

    class SafeNeighborCmd(Command):
        """Safe command that runs without touching siblings."""

        name = "safe"

        def __call__(self, count: int = 1) -> int:
            nonlocal ran_safe
            ran_safe = True
            return count * 2

    # 1. Registration must NOT evaluate UntouchedTrackedCmd.__call__.__annotate__
    cli = Cli(prog="lazytool", commands=[UntouchedTrackedCmd, SafeNeighborCmd])
    assert annotate_evals == 0

    # 2. Root --help rendering (including 80-column CI boxed wrapping) must NOT evaluate __annotate__
    monkeypatch.delenv("CLIO_AGENT", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "80")
    help_text = cli.format_help()
    assert "tracked (tr)" in help_text
    assert "Command whose PEP 649 __annotate__ tracks evaluation count." in _reflow_help(help_text)
    assert annotate_evals == 0

    # 3. Routing and running a sibling command must NOT evaluate UntouchedTrackedCmd.__call__.__annotate__
    assert cli.run(["safe", "--count", "4"]) == 8
    assert ran_safe is True
    assert annotate_evals == 0

    # 4. Invoking the tracked command evaluates __annotate__ exactly once and caches CommandSpec
    assert cli.run(["tracked", "10"]) == 11
    assert annotate_evals == 1
    assert cli.run(["tr", "20"]) == 21
    assert annotate_evals == 1


def test_pep695_type_aliases_and_nested_forward_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PEP695_WORKERS", "12")
    monkeypatch.setenv("CLIO_TEST_TOKEN", "secret-token-42")
    captured: dict[str, Any] = {}

    type OptionalPair[T] = tuple[T, T] | None

    class PipelineCommand(Command):
        """Run pipeline using PEP 695 type aliases and forward-referenced types.

        Args:
            workers: Docstring fallback description for workers.
        """

        name = "pipe"

        def __call__(
            self,
            stage: Annotated[LaterStageConfig, "Deferred stage config"],
            host: Annotated[str, Arg(positional=True, metavar="HOST"), "Bind hostname"] = "127.0.0.1",
            workers: WorkerCount = 4,
            fmt: OutputFormat = "json",
            tags: TaggedValues[int] = [],  # noqa: B006
            bounds: OptionalPair[float] = None,
            level: LaterLevelEnum | None = None,
            token: Annotated[str, Arg(short="-k", env="CLIO_TEST_TOKEN", help="Auth token")] = "",
            context: UnresolvedFutureType | None = None,
        ) -> None:
            captured.update(
                stage=stage,
                host=host,
                workers=workers,
                fmt=fmt,
                tags=tags,
                bounds=bounds,
                level=level,
                token=token,
                context=context,
            )

    @dataclass(frozen=True, slots=True)
    class LaterStageConfig:
        """Defined AFTER PipelineCommand to exercise nested PEP 649 ForwardRef resolution."""

        batch_size: int = 64

    class LaterLevelEnum(enum.Enum):
        LOW = "low"
        HIGH = "high"

    cli = Cli(prog="srv", commands=[PipelineCommand])
    cli.run(
        [
            "pipe",
            "--fmt",
            "parquet",
            "-t",
            "10",
            "20",
            "--bounds",
            "-1.5",
            "2.5",
            "--level",
            "high",
            "--batch-size",
            "128",
        ]
    )

    assert captured == {
        "stage": LaterStageConfig(batch_size=128),
        "host": "127.0.0.1",
        "workers": 12,
        "fmt": "parquet",
        "tags": [10, 20],
        "bounds": (-1.5, 2.5),
        "level": LaterLevelEnum.HIGH,
        "token": "secret-token-42",
        "context": None,
    }

    cli.run(["pipe", "0.0.0.0", "-w=32", "-k", "override"])
    assert captured["host"] == "0.0.0.0"
    assert captured["workers"] == 32
    assert captured["token"] == "override"

    schema = cli.to_json_schema()
    pipe_props = schema["commands"]["pipe"]["properties"]
    assert pipe_props["fmt"]["enum"] == ["json", "tsv", "parquet"]
    assert pipe_props["tags"]["items"] == {"type": "integer"}
    assert pipe_props["level"]["enum"] == ["low", "high"]

    md = cli.to_markdown()
    assert "`-w`, `--workers`" in md
    assert "[env: PEP695_WORKERS]" in md
    assert "Docstring fallback description for workers." in md
    assert "Auth token" in md
    assert "Bind hostname" in md


# 8. Native Scanner Strict Semantics, Tuples, Negatives & Single-Command Mode -----------------------------------------
def test_native_scanner_strict_semantics_tuples_negatives_and_bool_overrides(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    monkeypatch.setenv("FORCE_CACHE_ENV", "true")

    class Priority(enum.Enum):
        LOW = 1
        HIGH = 2

    class ComputeCmd(Command):
        """Run numeric & tuple parsing checks."""

        name = "compute"

        def __call__(
            self,
            label: str,
            offset: Annotated[float, Arg(positional=True)] = 0.0,
            *,
            delta: float = 0.0,
            coords: tuple[float, float] = (0.0, 0.0),
            endpoint: tuple[str, int] = ("localhost", 8080),
            nums: Annotated[list[int], Arg(short="-n")] = [],  # noqa: B006
            cache: Annotated[bool, Arg(env="FORCE_CACHE_ENV")] = False,
            priority: Priority = Priority.LOW,
        ) -> dict[str, Any]:
            return {
                "label": label,
                "offset": offset,
                "delta": delta,
                "coords": coords,
                "endpoint": endpoint,
                "nums": nums,
                "cache": cache,
                "priority": priority,
            }

    cli = Cli(prog="calc", commands=[ComputeCmd])

    # 1. Negative positional numbers, negative option/tuple/variadic values, repeated flags, --no-cache overriding env,
    #    --flag=val, Enum by member name, and `--` sentinel for leading-dash positional strings
    res = cli.run(
        [
            "compute",
            "--delta",
            "-9.5",
            "--coords",
            "-12.5",
            "-.25",
            "--endpoint",
            "db.local",
            "5432",
            "-n",
            "-1",
            "-2",
            "-n=3",
            "--no-cache",
            "--priority=HIGH",
            "--",
            "--dash-label",
            "-42.5",
        ]
    )
    assert res == {
        "label": "--dash-label",
        "offset": -42.5,
        "delta": -9.5,
        "coords": (-12.5, -0.25),
        "endpoint": ("db.local", 5432),
        "nums": [-1, -2, 3],
        "cache": False,
        "priority": Priority.HIGH,
    }

    # Negative positional float passed directly WITHOUT `--` sentinel
    res_neg_pos = cli.run(["compute", "plain", "-3.25"])
    assert res_neg_pos["offset"] == -3.25

    # 2. Explicit --cache=false also overrides truthy env variable
    res2 = cli.run(["compute", "item", "--cache=false"])
    assert res2["cache"] is False

    # 3. --no-cache=true is rejected
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["compute", "item", "--no-cache=true"])
    assert exc_info.value.code == 2
    assert "ignored explicit argument" in capsys.readouterr().err

    # 4. Wrong fixed-tuple arity raises ParseError
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["compute", "item", "--coords", "1.0"])
    assert exc_info.value.code == 2
    assert "expected 2 arguments" in capsys.readouterr().err

    # 5. Attached short value (-n5) is strictly rejected with fuzzy suggestion
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["compute", "item", "-n5"])
    assert exc_info.value.code == 2
    err_out = capsys.readouterr().err
    assert "unrecognized arguments: -n5" in err_out
    assert "Did you mean '-n'?" in err_out


def test_single_command_execution_and_hybrid_command_routing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)

    class SingleEcho(Command):
        """Standalone echo utility."""

        name = "echo"

        def __call__(self, text: str, *, times: Annotated[int, Arg(short="-t")] = 1) -> str:
            return " ".join([text] * times)

    # 1. Direct Command.run() convenience entry point
    assert SingleEcho.run(["hi", "-t", "3"]) == "hi hi hi"

    # 2. Single-command Cli(command=...) with version, help, markdown, schema, skill, dry-run, and MCP
    single_cli = Cli(
        prog="echoctl",
        version="3.1.0",
        epilog="Single mode epilog.",
        command=SingleEcho,
        enable_agent_flags=True,
    )
    assert "usage: echoctl [options] TEXT" in single_cli.format_help()
    assert "# `echoctl`" in single_cli.to_markdown()
    assert single_cli.to_json_schema()["commands"]["echo"]["required"] == ["text"]
    assert "name: echoctl" in single_cli.to_skill_md()

    dry = single_cli.run(["hello", "-t", "2", "--dry-run"])
    assert isinstance(dry, AgentResult)
    assert dry.arguments == {"text": "hello", "times": 2}
    capsys.readouterr()

    mcp_res = McpServer(single_cli).call_tool("echo", {"text": "mcp", "times": 2})
    assert mcp_res.ok is True
    assert mcp_res.result == "mcp mcp"

    with pytest.raises(ValueError, match="Cannot specify both"):
        Cli(command=SingleEcho, commands=[LeafCommand])
    with pytest.raises(ValueError, match="Cannot add subcommands"):
        single_cli.add_command(LeafCommand)

    # 3. Hybrid Command (both executable directly AND hosting child subcommands)
    class ServiceStatusCmd(Command):
        """Inspect service status."""

        name = "status"

        def __call__(self) -> str:
            return "healthy"

    class ServiceHybridCmd(Command):
        """Run service or a service subcommand."""

        name = "service"
        subcommands = [ServiceStatusCmd]

        def __call__(self, port: int = 8080) -> str:
            return f"serving:{port}"

    hybrid_cli = Cli(prog="svc", commands=[ServiceHybridCmd])
    assert hybrid_cli.run(["service", "--port", "9000"]) == "serving:9000"
    assert hybrid_cli.run(["service", "status"]) == "healthy"


# 9. Agentic Coding, Schema, SKILL.md, Dry-Run & MCP Tests ------------------------------------------------------------
def test_schema_tool_and_skill_generation() -> None:
    class DeployCommand(Command):
        """Deploy service to target environment.

        Args:
            target: Path to deployment manifest.
            env: Target environment name.
            replicas: Number of replicas to provision.
        """

        name = "deploy"
        aliases = ["dp"]

        def __call__(
            self,
            target: Path,
            env: Literal["staging", "prod"] = "staging",
            replicas: Annotated[int, Arg(short="-r", env="DEPLOY_REPLICAS")] = 2,
        ) -> dict[str, Any]:
            return {"target": target, "env": env, "replicas": replicas}

    cli = Cli(
        prog="cloudctl",
        description="Cloud control plane CLI.",
        version="2.0.0",
        commands=[DeployCommand, ParentBranchCommand],
        enable_agent_flags=True,
    )

    schema = cli.to_json_schema()
    assert schema["prog"] == "cloudctl"
    assert schema["version"] == "2.0.0"
    assert "deploy" in schema["commands"]
    assert "branch" in schema["commands"]

    deploy_schema = schema["commands"]["deploy"]
    assert deploy_schema["required"] == ["target"]
    assert deploy_schema["properties"]["target"]["format"] == "path"
    assert deploy_schema["properties"]["env"]["enum"] == ["staging", "prod"]
    assert deploy_schema["properties"]["replicas"]["type"] == "integer"
    assert deploy_schema["properties"]["replicas"]["x-clio-env"] == "DEPLOY_REPLICAS"
    assert "-r" in deploy_schema["properties"]["replicas"]["x-clio-flags"]

    tools = cli.to_tools()
    tool_names = [t.name for t in tools]
    assert tool_names == ["deploy", "branch_leafcommand"]
    assert isinstance(tools[0], ToolSpec)
    assert tools[0].to_openai_tool()["type"] == "function"

    skill_md = cli.to_skill_md()
    assert skill_md.startswith("---\nname: cloudctl\n")
    assert "CLIO_AGENT=1" in skill_md
    assert "--json" in skill_md
    assert "--dry-run" in skill_md
    assert "--mcp" in skill_md


def test_clio_agent_env_mode_concise_errors_and_no_spinners(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    monkeypatch.setenv("CLIO_AGENT", "1")

    class WorkCommand(Command):
        """Run work items."""

        name = "work"

        def __call__(self, count: int = 2) -> None:
            assert self.cli is not None
            for _ in self.cli.progress(range(count), "Working"):
                pass

    cli = Cli(prog="agenttool", commands=[WorkCommand])
    assert cli.agent_mode is True

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["work", "--coun", "5"])
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Did you mean '--count'?" in stderr
    assert "Run 'agenttool work --help' for full help." in stderr
    assert "🌎 Global options" not in stderr

    cli.run(["-V", "work", "--count", "3"])
    stderr_progress = capsys.readouterr().err
    assert "Working (3)" in stderr_progress
    assert "\r" not in stderr_progress


def test_opt_in_agent_flags_json_dry_run_and_introspection(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    executed: list[Any] = []

    @dataclass(frozen=True, slots=True)
    class BuildResult:
        artifact: Path
        replicas: int

    class BuildCommand(Command):
        """Build an artifact."""

        name = "build"

        def __call__(self, target: Path, *, replicas: int = 2, fail: bool = False) -> BuildResult:
            assert self.cli is not None
            self.cli.msg("Starting build")
            print("build-stdout-line")
            if fail:
                raise CliError("Build failed cleanly", exit_code=9)
            executed.append((target, replicas))
            return BuildResult(artifact=target, replicas=replicas)

    cli = Cli(prog="builder", version="1.0.0", commands=[BuildCommand], enable_agent_flags=True)

    dry_res = cli.run(["build", "dist/app.whl", "--replicas", "4", "--dry-run"])
    assert isinstance(dry_res, AgentResult)
    assert dry_res.ok is True
    assert dry_res.dry_run is True
    assert executed == []
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["dry_run"] is True
    assert dry_payload["arguments"] == {"target": "dist/app.whl", "replicas": 4, "fail": False}

    ok_res = cli.run(["-V", "build", "dist/app.whl", "--replicas", "3", "--json"])
    assert isinstance(ok_res, AgentResult)
    assert len(executed) == 1
    ok_payload = json.loads(capsys.readouterr().out)
    assert ok_payload["ok"] is True
    assert ok_payload["command"] == "build"
    assert ok_payload["result"] == {"artifact": "dist/app.whl", "replicas": 3}
    assert {"level": "info", "text": "Starting build"} in ok_payload["messages"]
    assert {"level": "stdout", "text": "build-stdout-line"} in ok_payload["messages"]

    assert ok_res.messages[0] == AgentMessage(level="info", text="Starting build")
    assert ok_res.error is None
    err_spec = AgentErrorSpec(message="Build failed cleanly", exit_code=9)
    assert err_spec.to_dict() == {"message": "Build failed cleanly", "exit_code": 9}

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["build", "dist/app.whl", "--fail", "--json"])
    assert exc_info.value.code == 9
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["ok"] is False
    assert err_payload["error"]["message"] == "Build failed cleanly"
    assert err_payload["error"]["exit_code"] == 9

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["--schema"])
    assert exc_info.value.code == 0
    schema_out = json.loads(capsys.readouterr().out)
    assert schema_out["prog"] == "builder"

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["build", "--skill"])
    assert exc_info.value.code == 0
    assert "name: builder-build" in capsys.readouterr().out


def test_mcp_server_json_rpc_lifecycle() -> None:
    @dataclass(frozen=True, slots=True)
    class SumTask:
        """Add two integers.

        Attributes:
            a: First number.
            b: Second number.
        """

        name: str = field(default="sum", init=False)
        a: int
        b: int = 10

        def __call__(self) -> dict[str, int]:
            return {"total": self.a + self.b}

    cli = Cli(prog="calc", version="0.2.0", commands=[SumTask, ParentBranchCommand])
    server = McpServer(cli)

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "sum", "arguments": {"a": 32, "b": 10}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "branch_leafcommand", "arguments": {"name": "agent"}},
        },
    ]
    in_stream = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    out_stream = io.StringIO()

    server.serve(stdin=in_stream, stdout=out_stream)
    responses = [json.loads(line) for line in out_stream.getvalue().splitlines() if line.strip()]

    assert len(responses) == 4
    assert responses[0]["result"]["serverInfo"]["name"] == "calc"
    listed_names = [t["name"] for t in responses[1]["result"]["tools"]]
    assert listed_names == ["sum", "branch_leafcommand"]

    sum_res = responses[2]["result"]
    assert sum_res["isError"] is False
    assert sum_res["structuredContent"]["result"] == {"total": 42}

    leaf_res = responses[3]["result"]
    assert leaf_res["isError"] is False
    assert {"level": "stdout", "text": "Hello, agent!"} in leaf_res["structuredContent"]["messages"]


# 10. Opt-In Async Execution & Spinner Tests ---------------------------------------------------------------------------
def test_async_commands_run_and_run_async_and_mcp(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class AsyncFetchCommand(Command):
        """Fetch remote items concurrently."""

        name = "fetch"

        async def __call__(self, urls: list[str], *, fail: bool = False) -> list[str]:
            async def _one(u: str) -> str:
                await asyncio.sleep(0)
                return u.upper()

            if fail:
                raise CliError("Async fetch failed", exit_code=17)

            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(_one(u)) for u in urls]
            return [t.result() for t in tasks]

    @dataclass(frozen=True, slots=True)
    class AsyncJobSpec:
        """Async callable dataclass command."""

        name: str = field(default="job", init=False)
        item: str

        async def __call__(self) -> str:
            await asyncio.sleep(0)
            return f"done:{self.item}"

    assert AsyncFetchCommand.get_spec().call_plan.is_async is True
    assert LeafCommand.get_spec().call_plan.is_async is False

    cli = Cli(prog="asynctool", commands=[AsyncFetchCommand, AsyncJobSpec], enable_agent_flags=True)

    # 1. Synchronous Cli.run() lazily runs coroutine via asyncio.run()
    assert cli.run(["fetch", "a", "b"]) == ["A", "B"]
    assert cli.run(["job", "payload"]) == "done:payload"

    # 2. --json mode with async command (success and caught CliError)
    ok_env = cli.run(["fetch", "x", "--json"])
    assert isinstance(ok_env, AgentResult)
    assert ok_env.result == ["X"]
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["fetch", "x", "--fail", "--json"])
    assert exc_info.value.code == 17
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["error"]["message"] == "Async fetch failed"

    # 3. McpServer.call_tool() (sync) on async command
    mcp = McpServer(cli)
    sync_tool_res = mcp.call_tool("fetch", {"urls": ["m1", "m2"]})
    assert sync_tool_res.ok is True
    assert sync_tool_res.result == ["M1", "M2"]

    # 4. Native async host execution via Cli.run_async, Command.run_async, and McpServer.call_tool_async
    async def _exercise_async_apis() -> None:
        r1 = await cli.run_async(["fetch", "p", "q"])
        assert r1 == ["P", "Q"]

        r2 = await cli.run_async(["fetch", "z", "--json"])
        assert isinstance(r2, AgentResult)
        assert r2.result == ["Z"]

        r3 = await AsyncFetchCommand.run_async(["solo"])
        assert r3 == ["SOLO"]

        r4 = await mcp.call_tool_async("fetch", {"urls": ["k"]})
        assert r4.ok is True
        assert r4.result == ["K"]

        r5 = await mcp.call_tool_async("unknown_tool", {})
        assert r5.ok is False

    asyncio.run(_exercise_async_apis())
    capsys.readouterr()


def test_spinner_sync_async_tty_agent_json_and_decorator(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    monkeypatch.setattr(Colors, "ENABLED", False)
    monkeypatch.delenv("CLIO_AGENT", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    class FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    # 1. Interactive TTY animation, .update(), .succeed(), and cursor hide/restore
    tty_buf = FakeTTY()
    fast_cfg = SpinnerConfig(frames=("-", "\\", "|", "/"), interval=0.005)
    with Spinner("Loading", config=fast_cfg, stream=tty_buf, agent_mode=False) as sp:
        time.sleep(0.02)
        sp.update("Finalizing")
        time.sleep(0.01)
        sp.succeed("Completed cleanly")

    tty_out = tty_buf.getvalue()
    assert "\033[?25l" in tty_out  # Cursor hidden on enter
    assert "\033[?25h" in tty_out  # Cursor restored on exit
    assert "✔ Completed cleanly" in tty_out

    # 2. Exception inside interactive TTY spinner still restores cursor and emits error glyph
    tty_err_buf = FakeTTY()
    with pytest.raises(ValueError, match="boom"):
        with Spinner("Working", config=fast_cfg, stream=tty_err_buf, agent_mode=False):
            raise ValueError("boom")
    err_tty_out = tty_err_buf.getvalue()
    assert "\033[?25h" in err_tty_out
    assert "✖ Working" in err_tty_out

    # 3. Agent / non-TTY mode disables \r and cursor codes even on FakeTTY, and supports .fail()
    agent_buf = FakeTTY()
    with Spinner("Checking", stream=agent_buf, agent_mode=True, show_elapsed=False) as sp2:
        sp2.fail("Check failed")
    agent_out = agent_buf.getvalue()
    assert "\033[?25l" not in agent_out
    assert "\r" not in agent_out
    assert agent_out == "✖ Check failed\n"

    # 4. @spinner decorator on sync function, async function, and Command.__call__ + --json capture
    @spinner("Sync step", show_elapsed=False)
    def compute_double(x: int) -> int:
        return x * 2

    @spinner("Async step", show_elapsed=False)
    async def async_triple(x: int) -> int:
        await asyncio.sleep(0)
        return x * 3

    non_tty_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", non_tty_err)
    assert compute_double(21) == 42
    assert asyncio.run(async_triple(7)) == 21
    assert "✔ Sync step" in non_tty_err.getvalue()
    assert "✔ Async step" in non_tty_err.getvalue()

    class SpinCommand(Command):
        """Command decorated with @spinner that also uses cli.spinner().

        Args:
            target: Target artifact name.
        """

        name = "spin"

        @spinner("Building target", show_elapsed=False)
        async def __call__(self, target: str) -> str:
            assert self.cli is not None
            async with self.cli.spinner("Uploading target", show_elapsed=False) as sub_sp:
                await asyncio.sleep(0)
                sub_sp.succeed(f"Uploaded {target}")
            return target

    # Verify @spinner preserved __wrapped__ so DocstringParser & ScanPlan see `target`!
    spin_specs = SpinCommand().get_parameter_specs()
    assert [s.name for s in spin_specs] == ["target"]
    assert spin_specs[0].docstring == "Target artifact name."

    spin_cli = Cli(prog="spintool", commands=[SpinCommand], enable_agent_flags=True)
    json_res = spin_cli.run(["spin", "wheel.whl", "--json"])
    assert isinstance(json_res, AgentResult)
    assert json_res.ok is True
    assert json_res.result == "wheel.whl"
    msg_texts = [m.text for m in json_res.messages]
    assert "Uploaded wheel.whl" in msg_texts
    assert "Building target" in msg_texts
    capsys.readouterr()


# 11. Extended Branch Coverage Across Modules --------------------------------------------------------------------------
def test_extended_coverage_color_docstr_parser_command_agent_and_cli(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clio.color import _detect_color_support
    from clio.docstr import _parse_bool

    # 1. Color environment detection & style branches
    monkeypatch.setenv("NO_COLOR", "1")
    assert _detect_color_support() is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _detect_color_support() is True
    monkeypatch.delenv("FORCE_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert _detect_color_support() is False
    monkeypatch.delenv("TERM")
    assert isinstance(_detect_color_support(), bool)

    monkeypatch.setattr(Colors, "ENABLED", True)
    full_style = Style(
        fg=Colors.RED,
        bg=Colors.BG_BLUE,
        bold=True,
        dim=True,
        italic=True,
        underline=True,
        blink=True,
        reverse=True,
        hidden=True,
        strikethrough=True,
    )
    assert full_style("x").startswith("\033[")
    assert full_style("") == ""
    assert Colors.wrap("x", "") == "x"
    assert Colors.style("x") == "x"
    assert Colors.style(
        "x",
        fg=Colors.RED,
        bg=Colors.BG_BLACK,
        bold=True,
        dim=True,
        italic=True,
        underline=True,
        blink=True,
        reverse=True,
        hidden=True,
        strikethrough=True,
    ).startswith("\033[")
    assert Colors.black("b") != "b"
    assert Colors.black("b", bold=True) != "b"
    assert Colors.green("g") != "g"
    assert Colors.yellow("y", bold=True) != "y"
    assert Colors.blue("bl", bold=True) != "bl"
    assert Colors.magenta("m", bold=True) != "m"
    assert Colors.cyan("c") != "c"
    assert Colors.white("w", bold=True) != "w"
    assert Colors.gray("gr", bold=True) != "gr"
    with pytest.raises(ValueError, match="256-color"):
        Colors.bg256(-1)
    with pytest.raises(ValueError, match="Invalid hex"):
        Colors.hex("#12345")

    # 2. _parse_bool, serialize_for_json, ParameterSpec coercion & schema
    assert _parse_bool("yes") is True
    assert _parse_bool("off") is False
    with pytest.raises(ValueError, match="invalid boolean value"):
        _parse_bool("maybe")

    class ColorEnum(enum.Enum):
        RED = "red"
        BLUE = "blue"

    class CustomDictObj:
        def to_dict(self) -> dict[str, Any]:
            return {"nested": ColorEnum.RED}

    class CustomFallbackObj:
        def __str__(self) -> str:
            return "custom-fallback"

    assert serialize_for_json(ColorEnum.BLUE) == "blue"
    assert serialize_for_json({"k": (1, 2)}) == {"k": [1, 2]}
    assert serialize_for_json(CustomDictObj()) == {"nested": "red"}
    assert serialize_for_json(CustomFallbackObj()) == "custom-fallback"

    a1 = Arg(short="-a", help="first")
    a2 = Arg(short=("-a", "--alpha"), metavar="A", choices=("x", "y"), env="ENV_A", positional=False)
    merged_arg = a1.merge(a2)
    assert merged_arg.short_flags == ("-a", "--alpha")
    assert merged_arg.metavar == "A"

    assert ParameterSpec(name="untyped", annotation=inspect.Parameter.empty).type_label == "str"

    p_bool = ParameterSpec(name="flag", annotation=bool, default=False)
    assert p_bool._coerce_env_value("true") is True
    assert p_bool.coerce_value("1") is True
    assert p_bool.coerce_value(None) is None

    p_list_int = ParameterSpec(name="nums", annotation=list[int])
    assert p_list_int._coerce_env_value("1, 2, 3") == [1, 2, 3]
    assert p_list_int.coerce_value("4,5") == [4, 5]

    p_list_lit = ParameterSpec(name="modes", annotation=list[Literal["a", "b"]])
    assert p_list_lit.coerce_value("a,b") == ["a", "b"]
    with pytest.raises(ValueError, match="Invalid choice"):
        p_list_lit.coerce_value(["a", "c"])

    p_tuple = ParameterSpec(name="tup", annotation=tuple[int, ...])
    assert p_tuple.coerce_value([1, "2"]) == (1, 2)

    p_fixed_tuple = ParameterSpec(name="pair", annotation=tuple[int, str])
    assert p_fixed_tuple.coerce_value("7,ok") == (7, "ok")
    with pytest.raises(ValueError, match="Expected 2 values"):
        p_fixed_tuple.coerce_value(["7"])

    p_set = ParameterSpec(name="st", annotation=set[str])
    assert p_set.coerce_value("x,y") == {"x", "y"}

    p_lit = ParameterSpec(name="lit", annotation=Literal[1, 2])
    assert p_lit.coerce_value("2") == 2
    with pytest.raises(ValueError, match="Invalid choice"):
        p_lit.coerce_value("9")

    assert ParameterSpec._annotation_to_json_schema(int | str) == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert ParameterSpec._annotation_to_json_schema(dict[str, int]) == {"type": "object"}
    assert ParameterSpec._annotation_to_json_schema(float) == {"type": "number"}
    assert ParameterSpec._annotation_to_json_schema(ColorEnum) == {"type": "string", "enum": ["red", "blue"]}
    assert ParameterSpec._annotation_to_json_schema(tuple[int, str])["prefixItems"] == [
        {"type": "integer"},
        {"type": "string"},
    ]
    DocstringParser.clear_cache()

    # 3. Positional-only dataclass & **kwargs CallPlan + invoke_with_arguments
    @dataclass
    class PosCfg:
        host: str = "localhost"

    class ComplexCallCmd(Command):
        """Exercise positional-only dataclass, positional scalar, and **kwargs CallPlan branches."""

        name = "imp"

        def __call__(
            self,
            target: str,
            cfg: PosCfg,
            /,
            *,
            mode: Literal["fast", "slow"] = "fast",
            items: list[int] = [1],  # noqa: B006
            **kwargs: Any,
        ) -> dict[str, Any]:
            if self.cli is not None:
                for _ in self.cli.progress([1, 2], "Iterating"):
                    pass
            print("err-line", file=sys.stderr)
            return {"host": cfg.host, "target": target, "mode": mode, "items": items, "kw": kwargs}

    imp_cmd = ComplexCallCmd()
    specs = imp_cmd.get_parameter_specs()
    assert any(s.name == "target" for s in specs)
    assert imp_cmd.invoke_with_arguments({"target": "app", "extra_k": 9})["target"] == "app"
    assert asyncio.run(imp_cmd.invoke_with_arguments_async({"target": "app"}))["mode"] == "fast"

    # Exercise ScanPlan/OptionBinding/ScanResult/ParseError directly
    assert isinstance(imp_cmd.get_scan_plan().flag_map["--mode"], OptionBinding)
    scan_res = imp_cmd.get_scan_plan().parse(["app", "--mode", "slow"], prog="imp")
    assert isinstance(scan_res, ScanResult)
    assert scan_res.values["mode"] == "slow"
    pe = ParseError("bad flag", usage="usage: imp", suggestions=("--mode",), command_path="imp")
    assert pe.exit_code == 2

    with pytest.raises(ValueError, match="cannot be null"):
        imp_cmd.invoke_with_arguments({"target": None})
    with pytest.raises(ValueError, match="Missing required argument"):
        imp_cmd.invoke_with_arguments({})
    with pytest.raises(TypeError, match="not executable"):
        ParentBranchCommand().invoke_with_arguments({})
    with pytest.raises(TypeError, match="Expected a BaseCommand"):
        Cli(prog="bad", commands=[123])  # type: ignore
    with pytest.raises(TypeError, match="Expected a @dataclass"):
        Command.from_dataclass(int)
    with pytest.raises(TypeError, match="must either define '__call__'"):
        Command.from_dataclass(PosCfg)
    with pytest.raises(TypeError, match="must either define '__call__'"):
        Cli(prog="bad", commands=[PosCfg])

    # 4. Custom Cli subclass, --json parser error, invalid choice fuzzy match, --help-md, --schema, --skill, and MCP errors
    class CustomCli(Cli):
        """Custom CLI subclass docstring."""

    app = CustomCli(prog="custom", version="0.1.0", commands=[imp_cmd], enable_agent_flags=True)
    assert app.config.description == "Custom CLI subclass docstring."

    dry_imp = app.run(["imp", "my-target", "--dry-run"])
    assert dry_imp.dry_run is True
    capsys.readouterr()

    json_imp = app.run(["-V", "imp", "my-target", "--json"])
    assert json_imp.ok is True
    assert any(m.level == "stderr" and m.text == "err-line" for m in json_imp.messages)
    assert any(m.level == "info" and "Iterating (2)" in m.text for m in json_imp.messages)
    capsys.readouterr()

    # Parser error with --json emits structured JSON error envelope
    with pytest.raises(SystemExit) as exc_info:
        app.run(["imp", "my-target", "--mod", "fast", "--json"])
    assert exc_info.value.code == 2
    err_json = json.loads(capsys.readouterr().out)
    assert err_json["ok"] is False
    assert "--mode" in err_json["error"]["suggestions"]

    # Invalid choice fuzzy suggestion
    with pytest.raises(SystemExit) as exc_info:
        app.run(["imp", "my-target", "--mode", "fas", "--json"])
    assert exc_info.value.code == 2
    choice_err = json.loads(capsys.readouterr().out)
    assert "fast" in choice_err["error"]["suggestions"]

    # Missing required argument error
    with pytest.raises(SystemExit) as exc_info:
        app.run([])
    assert exc_info.value.code == 2
    capsys.readouterr()

    # Empty CLI with no commands prints help cleanly and returns None
    empty_cli = Cli(prog="empty")
    assert empty_cli.run([]) is None
    assert "usage: empty" in Colors.strip_ansi(capsys.readouterr().out)

    # --help, --help-md, --schema, --skill on root and subcommand
    with pytest.raises(SystemExit):
        app.run(["--help"])
    assert "usage: custom" in Colors.strip_ansi(capsys.readouterr().out)

    with pytest.raises(SystemExit):
        app.run(["imp", "--help"])
    assert "usage: custom imp" in Colors.strip_ansi(capsys.readouterr().out)

    with pytest.raises(SystemExit):
        app.run(["--help-md"])
    assert "# `custom`" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        app.run(["imp", "--help-md"])
    assert "## `custom imp`" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        app.run(["imp", "--schema"])
    assert json.loads(capsys.readouterr().out)["title"] == "imp"

    with pytest.raises(SystemExit):
        app.run(["--skill"])
    assert "name: custom" in capsys.readouterr().out

    # MCP error handling branches
    mcp_lines = [
        "",
        "{invalid-json",
        "[1, 2, 3]",
        json.dumps({"jsonrpc": "2.0", "id": 10, "method": "ping"}),
        json.dumps({"jsonrpc": "2.0", "id": 11, "method": "unknown/method"}),
        json.dumps({"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {"name": "imp", "arguments": "bad"}}),
        json.dumps(
            {"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {"name": "no_such_tool", "arguments": {}}}
        ),
        json.dumps({"jsonrpc": "2.0", "id": 15, "method": "tools/call", "params": {"name": "imp", "arguments": {}}}),
    ]
    mcp_in = io.StringIO("\n".join(mcp_lines) + "\n")
    mcp_out = io.StringIO()
    app.serve_mcp(stdin=mcp_in, stdout=mcp_out)
    mcp_resps = [json.loads(line) for line in mcp_out.getvalue().splitlines() if line.strip()]
    assert mcp_resps[0]["error"]["code"] == -32700
    assert mcp_resps[1]["error"]["code"] == -32600
    assert mcp_resps[2]["result"] == {}
    assert mcp_resps[3]["error"]["code"] == -32601
    assert mcp_resps[4]["error"]["code"] == -32602
    assert mcp_resps[5]["error"]["code"] == -32602
    assert mcp_resps[6]["result"]["isError"] is True
    assert mcp_resps[7]["result"]["isError"] is True


def test_scanner_formatter_group_and_async_error_branches(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)

    class StageEnum(enum.Enum):
        DEV = "dev"
        PROD = "prod"

    @dataclass
    class SelfDataclassCmd(Command):
        """Command class that is itself a @dataclass."""

        name: str = field(default="selfdc", init=False)
        item: str
        retries: int = 2

        def __call__(self) -> tuple[str, int]:
            return (self.item, self.retries)

    classPositionalEdges = None  # marker

    class PositionalEdgesCmd(Command):
        """Exercise variadic + fixed-tuple positionals, env fallbacks, and long option wrapping.

        Args:
            sources: Source files list.
            pair: Fixed coordinate pair.
            very_long_descriptive_configuration_flag: First paragraph of help.

                Second paragraph that is intentionally long enough to trigger line wrapping inside HelpFormatter.
        """

        name = "posedges"

        def __call__(
            self,
            sources: list[str],
            pair: tuple[int, str],
            *,
            tags: set[str] = set(),  # noqa: B006
            meta: dict[str, str] = {},  # noqa: B006
            stages: list[StageEnum] = [],  # noqa: B006
            flag_bool: Literal[True, False] = False,
            very_long_descriptive_configuration_flag: Annotated[str, Arg(env="LONG_CFG_ENV")] = "default",
            req_opt: list[int],
        ) -> dict[str, Any]:
            assert classPositionalEdges is None
            return {
                "sources": sources,
                "pair": pair,
                "tags": tags,
                "meta": meta,
                "stages": stages,
                "flag_bool": flag_bool,
                "long_cfg": very_long_descriptive_configuration_flag,
                "req_opt": req_opt,
            }

    monkeypatch.setenv("LONG_CFG_ENV", "from-env")
    cli = Cli(
        prog="edgetool",
        version="1.0.0",
        commands=[PositionalEdgesCmd, SelfDataclassCmd, ParentBranchCommand],
        formatter=HelpFormatter(max_width=65, col_width=24),
        enable_agent_flags=True,
    )

    # 1. Variadic positional followed by fixed-tuple positional + Enum list + Literal[bool] + long help wrapping
    help_str = PositionalEdgesCmd().format_help(
        prog_prefix="edgetool",
        formatter=HelpFormatter(max_width=65, col_width=24),
    )
    assert "--very-long-descriptive-configuration-flag" in help_str
    assert "[env: LONG_CFG_ENV]" in " ".join(help_str.split())

    res = cli.run(
        [
            "posedges",
            "f1.txt",
            "f2.txt",
            "7",
            "ok",
            "--stages",
            "dev",
            "prod",
            "--flag-bool",
            "true",
            "--req-opt",
            "10",
            "20",
        ]
    )
    assert res["sources"] == ["f1.txt", "f2.txt"]
    assert res["pair"] == (7, "ok")
    assert res["stages"] == [StageEnum.DEV, StageEnum.PROD]
    assert res["flag_bool"] is True
    assert res["long_cfg"] == "from-env"
    assert res["req_opt"] == [10, 20]

    # 2. Self-@dataclass Command execution + --dry-run resolve_arguments
    assert cli.run(["selfdc", "alpha", "--retries", "5"]) == ("alpha", 5)
    dry_selfdc = cli.run(["--dry-run", "selfdc", "beta"])
    assert isinstance(dry_selfdc, AgentResult)
    assert dry_selfdc.arguments == {"item": "beta", "retries": 2}
    capsys.readouterr()

    # 3. Scanner ParseError edge cases (missing scalar arg, invalid type, invalid bool, missing variadic opt, bad tuple pos)
    for bad_args in (
        ["--unknown-root-flag"],
        ["--", "unknown_cmd"],
        ["branch", "--unknown-group-flag"],
        ["branch", "leafcomman"],
        ["selfdc", "alpha", "--retries"],
        ["selfdc", "alpha", "--retries", "--other"],
        ["selfdc", "alpha", "--retries", "not-an-int"],
        ["posedges", "only_one_token", "--req-opt", "1"],
        ["posedges", "a", "7", "ok", "--req-opt"],
        ["posedges", "a", "7", "ok", "--req-opt="],
        ["posedges", "a", "7", "ok", "--req-opt", "1", "--stages", "invalid_stage"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            cli.run(bad_args)
        assert exc_info.value.code == 2
        capsys.readouterr()

    # 4. CommandGroup eager introspection (--help, --schema, --help-md, --skill) and resolve_arguments fallback
    assert ParentBranchCommand().resolve_arguments({"x": 1}) == {"x": 1}
    for group_eager in (
        ["branch", "--help"],
        ["branch", "--schema"],
        ["branch", "--help-md"],
        ["branch", "--skill"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            cli.run(group_eager)
        assert exc_info.value.code == 0
        capsys.readouterr()

    # 5. Single-command eager actions (--help, --version, --schema, --help-md, --skill)
    single_cli = Cli(prog="solo", version="0.9.0", command=SelfDataclassCmd, enable_agent_flags=True)
    for solo_eager in (["--help"], ["--version"], ["--schema"], ["--help-md"], ["--skill"]):
        with pytest.raises(SystemExit) as exc_info:
            single_cli.run(solo_eager)
        assert exc_info.value.code == 0
        capsys.readouterr()

    # 6. McpServer catching CliError in sync call_tool and async call_tool_async, plus run_async sync & error paths
    class ErrCmd(Command):
        """Raise CliError on demand."""

        name = "errcmd"

        async def __call__(self, *, code: int = 5) -> str:
            raise CliError("async boom", exit_code=code)

    err_cli = Cli(prog="errtool", commands=[ErrCmd, SelfDataclassCmd], enable_agent_flags=True)
    err_mcp = McpServer(err_cli)
    sync_err_tool = err_mcp.call_tool("errcmd", {"code": 6})
    assert sync_err_tool.ok is False
    assert sync_err_tool.error is not None
    assert sync_err_tool.error.exit_code == 6

    async def _async_checks() -> None:
        # Sync command via run_async
        assert await err_cli.run_async(["selfdc", "sync-in-async"]) == ("sync-in-async", 2)

        # Dry-run via run_async
        dr = await err_cli.run_async(["selfdc", "dry", "--dry-run"])
        assert isinstance(dr, AgentResult) and dr.dry_run is True

        # Async CliError in --json mode via run_async
        with pytest.raises(SystemExit) as e1:
            await err_cli.run_async(["errcmd", "--code", "11", "--json"])
        assert e1.value.code == 11

        # Async CliError in plain mode via run_async
        with pytest.raises(SystemExit) as e2:
            await err_cli.run_async(["errcmd", "--code", "12"])
        assert e2.value.code == 12

        # Async MCP call_tool_async error trapping
        async_err_tool = await err_mcp.call_tool_async("errcmd", {"code": 7})
        assert async_err_tool.ok is False
        assert async_err_tool.error is not None and async_err_tool.error.exit_code == 7

    asyncio.run(_async_checks())
    capsys.readouterr()

    # 7. DocstringParser.resolve_annotations direct checks
    hints = DocstringParser.resolve_annotations(SelfDataclassCmd)
    assert hints["item"] is str
    err_dict = AgentErrorSpec(message="bad", exit_code=2, usage="usage: x", suggestions=("--y",)).to_dict()
    assert err_dict["usage"] == "usage: x"
    assert err_dict["suggestions"] == ["--y"]


def test_plain_functions_first_class_cli_and_lazy_annotations(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)

    @dataclass(frozen=True, slots=True)
    class StepParams:
        """Hyperparameters for a training step.

        Attributes:
            lr: Learning rate.
            batch_size: Mini-batch size.
        """

        lr: float = 0.01
        batch_size: int = 32

    def run_step(cfg: StepParams, /, *, epochs: int = 3, tag: str = "exp") -> dict[str, Any]:
        """Run synchronous optimization step.

        Args:
            epochs: Number of passes.
            tag: Run tag label.
        """
        return {"lr": cfg.lr, "batch_size": cfg.batch_size, "epochs": epochs, "tag": tag}

    async def async_score(values: list[float], *, scale: float = 2.0) -> float:
        """Compute scaled sum asynchronously."""
        await asyncio.sleep(0)
        return sum(values) * scale

    broken_evals = 0

    def _raise_if_evaluated() -> type[int]:
        nonlocal broken_evals
        broken_evals += 1
        raise RuntimeError("sibling function __annotate__ must stay lazy!")

    def broken_sibling(x: _raise_if_evaluated()) -> int:  # type: ignore
        """Sibling function with failing PEP 649 annotation."""
        return x

    cli = Cli(
        prog="fnapp",
        commands=[run_step, async_score, broken_sibling],
        enable_agent_flags=True,
    )

    # Registration and root `--help` must NOT evaluate sibling function annotations
    assert broken_evals == 0
    root_help = cli.format_help()
    assert "run_step" in root_help
    assert "Run synchronous optimization step." in _reflow_help(root_help)
    assert broken_evals == 0

    # Both `run_step` and automatic kebab-case alias `run-step` route in O(1) without touching broken_sibling
    res1 = cli.run(["run_step", "--lr", "0.05", "--epochs", "5"])
    assert res1 == {"lr": 0.05, "batch_size": 32, "epochs": 5, "tag": "exp"}
    res2 = cli.run(["run-step", "--batch-size", "64", "--tag", "v2"])
    assert res2 == {"lr": 0.01, "batch_size": 64, "epochs": 3, "tag": "v2"}
    assert broken_evals == 0

    # `--dry-run` hydrates nested @dataclass bindings on plain functions
    dry = cli.run(["--dry-run", "run-step", "--lr", "0.1"])
    assert isinstance(dry, AgentResult)
    assert dry.dry_run is True
    assert dry.arguments["cfg"] == StepParams(lr=0.1, batch_size=32)
    capsys.readouterr()

    # Async plain function works via sync `cli.run()` and `cli.run_async()`
    assert cli.run(["async-score", "1.5", "2.5", "--scale", "3.0"]) == 12.0
    assert asyncio.run(cli.run_async(["async_score", "2.0", "4.0"])) == 12.0

    # Single-command `Cli(command=plain_fn)` and `Command.from_function()`
    fn_cmd = Command.from_function(run_step)
    assert isinstance(fn_cmd, FunctionCommand)
    assert fn_cmd(StepParams(lr=0.2, batch_size=8), epochs=1, tag="direct")["lr"] == 0.2
    assert fn_cmd.invoke_with_arguments({"lr": 0.3, "batch_size": 16})["batch_size"] == 16

    solo = Cli(command=async_score, enable_agent_flags=True)
    json_out = solo.run(["--json", "10", "5", "--scale", "0.5"])
    assert isinstance(json_out, AgentResult) and json_out.result == 7.5
    capsys.readouterr()

    with pytest.raises(TypeError):
        Command.from_function(StepParams)


def test_zero_edit_target_runner_and_dunder_main(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)

    # 1. TargetSpec parsing (POSIX, Windows drive letters, dotted modules, validation errors)
    s_file = TargetSpec.parse("tools/ analyze.py : run_job ")
    assert s_file == TargetSpec(module_or_path="tools/ analyze.py", symbol="run_job", is_file=True)
    s_win = TargetSpec.parse(r"C:\repo\script.py:my_fn")
    assert s_win == TargetSpec(module_or_path=r"C:\repo\script.py", symbol="my_fn", is_file=True)
    s_mod = TargetSpec.parse("pkg.submod:MyCls.method")
    assert s_mod == TargetSpec(module_or_path="pkg.submod", symbol="MyCls.method", is_file=False)
    s_whole = TargetSpec.parse("pkg.submod")
    assert s_whole == TargetSpec(module_or_path="pkg.submod", symbol=None, is_file=False)

    for invalid_spec in ("", "   ", ":only_sym", "script.py:"):
        with pytest.raises(CliError):
            TargetSpec.parse(invalid_spec)

    # 2. Untouched standalone external Python script (never imports `clio`, has `__main__` guard)
    ext_script = tmp_path / "ext_tools.py"
    ext_script.write_text(
        '''"""External repository data utilities."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelHyperParams:
    """Configuration schema without __call__.

    Attributes:
        lr: Learning rate.
        layers: Transformer block count.
    """

    lr: float = 1e-3
    layers: int = 12


@dataclass
class ExportTask:
    """Callable dataclass job."""

    dest: Path
    overwrite: bool = False

    def __call__(self) -> dict[str, Any]:
        return {"dest": str(self.dest), "overwrite": self.overwrite}


class ZeroArgCallable:
    """Plain zero-arg callable class."""

    def __call__(self, factor: int = 2) -> int:
        return factor * 21


class NeedsInitCallable:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def __call__(self) -> str:
        return self.secret


class PlainNotCallable:
    pass


CONSTANT_VALUE = 42


def summarize_records(
    input_path: Path,
    *,
    limit: int = 5,
    shout: bool = False,
) -> dict[str, Any]:
    """Summarize records from an input path.

    Args:
        input_path: Path to source file.
        limit: Max rows to inspect.
        shout: Uppercase summary label.
    """
    label = input_path.name.upper() if shout else input_path.name
    return {"label": label, "limit": limit}


async def async_pipeline(items: list[str], *, multiplier: int = 2) -> list[str]:
    """Run an async batch transformation."""
    await asyncio.sleep(0)
    return [item * multiplier for item in items]


def failing_task(reason: str = "bad state") -> None:
    """Raise a standard ValueError without importing clio."""
    raise ValueError(reason)


if __name__ == "__main__":
    raise RuntimeError("__main__ block must NOT execute when loaded by clio.runner!")
''',
        encoding="utf-8",
    )

    # 2a. Single-function `--dry-run --json`, live `--json`, and plain stdout via `runner_main` (`sys.exit(main())`-safe)
    assert runner_main([f"{ext_script}:summarize_records", "--dry-run", "--json", "data.csv", "--limit", "10"]) is None
    dry_stdout = json.loads(capsys.readouterr().out)
    assert dry_stdout["ok"] is True and dry_stdout["command"] == "summarize_records"
    assert dry_stdout["dry_run"] is True
    assert dry_stdout["arguments"] == {"input_path": "data.csv", "limit": 10, "shout": False}

    assert runner_main([f"{ext_script}:summarize_records", "--json", "notes.txt", "--shout"]) is None
    live_stdout = json.loads(capsys.readouterr().out)
    assert live_stdout["result"] == {"label": "NOTES.TXT", "limit": 5}

    assert runner_main([f"{ext_script}:summarize_records", "notes.txt", "--shout"]) is None
    assert json.loads(capsys.readouterr().out) == {"label": "NOTES.TXT", "limit": 5}

    assert runner_main([f"{ext_script}:ZeroArgCallable", "--factor", "3"]) is None
    assert capsys.readouterr().out.strip() == "63"

    # 2b. Async target via `run_target` and `run_target_async`
    assert run_target(f"{ext_script}:async_pipeline", ["ab", "cd", "--multiplier", "3"]) == ["ababab", "cdcdcd"]
    assert asyncio.run(run_target_async(f"{ext_script}:async_pipeline", ["x", "--multiplier", "4"])) == ["xxxx"]

    # 2c. Non-callable `@dataclass` target hydrated directly via `:ModelHyperParams`
    assert runner_main([f"{ext_script}:ModelHyperParams", "--lr", "0.02", "--json"]) is None
    assert json.loads(capsys.readouterr().out)["result"] == {"lr": 0.02, "layers": 12}

    # 2d. Zero-arg callable class works; non-callable / required-init / constant targets raise clean CliError
    assert run_target(f"{ext_script}:ZeroArgCallable", ["--factor", "3"]) == 63
    for bad_sym in ("NeedsInitCallable", "PlainNotCallable", "CONSTANT_VALUE"):
        with pytest.raises(CliError):
            build_target_cli(f"{ext_script}:{bad_sym}")

    # 2e. Domain exception inside untouched function is trapped cleanly in `--json` envelope
    with pytest.raises(SystemExit) as dom_exc:
        runner_main([f"{ext_script}:failing_task", "--reason", "disk full", "--json"])
    assert dom_exc.value.code == 1
    dom_json = json.loads(capsys.readouterr().out)
    assert dom_json["ok"] is False
    assert dom_json["error"]["message"] == "disk full"

    # 2f. Whole-file discovery (`ext_script` without `:symbol`): exposes public functions & callable @dataclass,
    #     skips `Path` import and non-callable `ModelHyperParams` DTO, and serves multi-tool `--mcp`!
    mod_cli = build_target_cli(str(ext_script))
    discovered_names = [cmd.name for cmd in mod_cli.commands]
    assert discovered_names == ["exporttask", "summarize_records", "async_pipeline", "failing_task"]

    mcp_requests = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "async_pipeline", "arguments": {"items": ["hi"], "multiplier": 2}},
            }
        ),
    ]
    mcp_in = io.StringIO("\n".join(mcp_requests) + "\n")
    mcp_out = io.StringIO()
    mod_cli.serve_mcp(stdin=mcp_in, stdout=mcp_out)
    mcp_lines = [json.loads(line) for line in mcp_out.getvalue().splitlines() if line.strip()]
    tool_names = [t["name"] for t in mcp_lines[0]["result"]["tools"]]
    assert "summarize_records" in tool_names and "async_pipeline" in tool_names
    assert json.loads(mcp_lines[1]["result"]["content"][0]["text"])["result"] == ["hihi"]

    # 3. Package hierarchy with `__init__.py` and relative imports + directory / dotted-module targets
    pkg_dir = tmp_path / "sample_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        '"""Sample package root."""\n\ndef root_ping(name: str = "world") -> str:\n    return f"hello {name}"\n',
        encoding="utf-8",
    )
    (pkg_dir / "constants.py").write_text("OFFSET = 100\n", encoding="utf-8")
    (pkg_dir / "ops.py").write_text(
        "from .constants import OFFSET\n\ndef add_offset(val: int) -> int:\n    return val + OFFSET\n",
        encoding="utf-8",
    )
    assert run_target(f"{pkg_dir / 'ops.py'}:add_offset", ["23"]) == 123
    assert run_target(str(pkg_dir), ["root-ping", "--name", "clio"]) == "hello clio"
    monkeypatch.chdir(tmp_path)
    assert run_target("sample_pkg.ops:add_offset", ["7"]) == 107

    # 4. Existing `Cli` instance in a module gets `enable_agent_flags=True` automatically
    app_file = tmp_path / "existing_cli_mod.py"
    app_file.write_text(
        "from clio import Cli\n\ndef greet(who: str = 'agent') -> str:\n    return f'hi {who}'\n\n"
        "app = Cli(prog='prebuilt', commands=[greet], enable_agent_flags=False)\n",
        encoding="utf-8",
    )
    assert runner_main([str(app_file), "greet", "--who", "tom", "--json"]) is None
    assert json.loads(capsys.readouterr().out)["result"] == "hi tom"

    # 5. Runner CLI top-level flags (`clio`, `--help`, `--version`, unknown flag, fuzzy symbol typo, `python -m clio`)
    assert runner_main([]) is None
    assert "usage: clio" in capsys.readouterr().out

    with pytest.raises(SystemExit) as h_exc:
        runner_main(["--help"])
    assert h_exc.value.code == 0
    assert "TARGET" in capsys.readouterr().out

    with pytest.raises(SystemExit) as v_exc:
        runner_main(["--version"])
    assert v_exc.value.code == 0
    assert capsys.readouterr().out.startswith("clio ")

    # Unknown pre-target flag in JSON mode
    with pytest.raises(SystemExit) as bad_opt_exc:
        runner_main(["--versio", "--json"])
    assert bad_opt_exc.value.code == 2
    bad_opt_json = json.loads(capsys.readouterr().out)
    assert "--version" in bad_opt_json["error"]["suggestions"]

    # Fuzzy symbol suggestion on `:typo` in JSON mode, agent mode, and human mode
    with pytest.raises(SystemExit) as sym_exc:
        runner_main([f"{ext_script}:summarize_record", "--json"])
    assert sym_exc.value.code == 2
    sym_json = json.loads(capsys.readouterr().out)
    assert "summarize_records" in sym_json["error"]["suggestions"]

    monkeypatch.setenv("CLIO_AGENT", "1")
    with pytest.raises(SystemExit):
        runner_main([f"{tmp_path / 'missing.py'}:fn"])
    assert "clio: error:" in capsys.readouterr().err

    monkeypatch.delenv("CLIO_AGENT", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    with pytest.raises(SystemExit):
        runner_main([f"{ext_script}:summarize_record"])
    runner_err = capsys.readouterr().err
    assert "Error" in runner_err
    assert "Did you mean" in runner_err and "summarize_records" in runner_err

    # Unboxed runner error fallback (`TERM=dumb`)
    monkeypatch.setenv("TERM", "dumb")
    with pytest.raises(SystemExit):
        runner_main([f"{ext_script}:summarize_record"])
    dumb_err = capsys.readouterr().err
    assert "Did you mean:" in dumb_err
    monkeypatch.setenv("TERM", "xterm-256color")

    # Execute `clio/__main__.py` directly via `runpy.run_module("clio", run_name="__main__")`
    monkeypatch.setattr(sys, "argv", ["clio", f"{ext_script}:summarize_records", "report.md", "--limit", "2", "--json"])
    runpy.run_module("clio", run_name="__main__")
    dunder_out = json.loads(capsys.readouterr().out)
    assert dunder_out["ok"] is True
    assert dunder_out["result"] == {"label": "report.md", "limit": 2}


def test_markdown_writer_primitives_and_edge_cases() -> None:
    """Verify MarkdownWriter GFM block builders, escaping, and empty no-op guards."""
    assert MarkdownWriter.escape_cell(None) == ""
    assert MarkdownWriter.escape_cell("") == ""
    assert MarkdownWriter.escape_cell("\x1b[31ma|b\nc\x1b[0m") == r"a\|b c"

    writer = (
        MarkdownWriter()
        .frontmatter({})
        .frontmatter({"name": "demo", "description": "Line 1\nLine 2"})
        .heading("Section", level=2)
        .paragraph(None)
        .paragraph("   ")
        .paragraph("Intro paragraph.")
        .bullets(["", "First item", "Second item"])
        .bullets([])
        .admonition("note", "Empty", ())
        .admonition("tip", "Fast Path", ("Zero sibling imports",))
        .raw(None)
        .raw("   ")
        .raw("> Blockquote")
        .table(("A", "B"), ())
        .table(
            ("Col A", "Col B"),
            [("x|y\nz", "42")],
            title="Metrics",
            alignments=("left", "right"),
            escape_cells=True,
        )
        .positional_table(())
        .option_table(())
        .command_table(())
        .rule()
    )
    rendered = str(writer)
    assert rendered == writer.render()
    assert rendered.startswith("---\nname: demo\ndescription: Line 1 Line 2\n---")
    assert "## Section" in rendered
    assert "- First item\n- Second item" in rendered
    assert '!!! tip "Fast Path"\n    Zero sibling imports' in rendered
    assert '!!! note "Empty"' not in rendered
    assert "> Blockquote" in rendered
    assert "**Metrics**\n\n| Col A | Col B |\n| :--- | ---: |\n| x\\|y z | 42 |" in rendered
    assert rendered.endswith("\n\n---")


def test_boxed_help_panels_arg_groups_and_box_primitive(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify BoxStyle, Colors.box/display_width/wrap_ansi, Arg(group=...), and agent/dumb auto-unboxing."""
    monkeypatch.delenv("CLIO_AGENT", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(Colors, "ENABLED", True)

    # 1. Unicode East-Asian / emoji display width & ANSI-preserving wrap/slice
    assert Colors.display_width("\x1b[1;36m💬 Commands\x1b[0m") == 11
    assert Colors.display_width("e\u0301\x07") == 1  # combining acute + control bell = 0 extra cells
    styled_long = Colors.cyan("A" * 28, bold=True) + " " + Colors.muted("[env: MY_LONG_VAR]")
    wrapped = Colors.wrap_ansi(styled_long, 20)
    assert len(wrapped) >= 3
    assert all(Colors.display_width(ln) <= 20 for ln in wrapped)
    assert any("[env: MY_LONG_VAR]" in Colors.strip_ansi(ln) for ln in wrapped)
    assert Colors.wrap_ansi("", 20) == [""]
    assert Colors.wrap_ansi("line1\n\nline2", 20) == ["line1", "", "line2"]

    # 2. Colors.box presets, alignments, title truncation, and exact uniform column widths
    for preset in (BoxStyle.ROUNDED, BoxStyle.SQUARE, BoxStyle.HEAVY, BoxStyle.DOUBLE, BoxStyle.ASCII):
        for align in ("left", "center", "right"):
            panel = Colors.box(
                [Colors.green("ready"), "🌎 status ok"],
                title="Very Long Panel Header Title",
                width=24,
                box_style=preset,
                border_style=Style(fg=Colors.CYAN, dim=True),
                title_style=Style(fg=Colors.CYAN, bold=True),
                align=align,
            )
            widths = {Colors.display_width(ln) for ln in panel.splitlines()}
            assert widths == {24}

    auto_panel = Colors.box("Auto-sized content", title="Info", max_width=60)
    assert len({Colors.display_width(ln) for ln in auto_panel.splitlines()}) == 1
    plain_box = Colors.box("No title box")
    assert plain_box.startswith("╭─")

    # 3. Arg(group=...) merge + boxed multi-column `--help` and grouped `--help-md`
    base_net = Arg(short="-p", group="Network")
    merged_net = base_net.merge(Arg(env="APP_PORT"))
    assert merged_net.group == "Network"
    assert merged_net.short_flags == ("-p",)
    assert merged_net.env == "APP_PORT"

    def deploy_cluster(
        bundle: Path,
        region: Annotated[str, Arg(group="Target")],
        host: Annotated[str, Arg(short="-H", group="Network")],
        port: Annotated[int, Arg(short="-p", env="APP_PORT", group="Network")] = 8443,
        token: Annotated[str, Arg(env="API_TOKEN", group="Auth", positional=False)] = "",
        mode: Literal["rolling", "canary"] = "rolling",
        dry_check: bool = False,
        labels: dict[str, str] = {},  # noqa: B006
        tags: list[int] = [],  # noqa: B006
        coords: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, Any]:
        """Deploy an artifact bundle to a remote cluster.

        Args:
            bundle: Local bundle archive path.
            region: Target deployment region.
            host: Remote cluster hostname.
            port: Remote HTTPS port.
            token: Bearer authentication token.
            mode: Rollout strategy.
            dry_check: Run preflight validation only.
            labels: Extra key=value metadata labels.
            tags: Numeric build tags.
            coords: Deployment coordinates.
        """
        return {"bundle": str(bundle), "region": region, "host": host, "port": port, "mode": mode}

    app = Cli(
        prog="deployctl",
        version="2.0.0",
        command=deploy_cluster,
        formatter=HelpFormatter(max_width=88),
        enable_agent_flags=True,
    )

    boxed_help = Colors.strip_ansi(app.format_help())
    assert "╭─ Positional arguments " in boxed_help
    assert "╭─ Options " in boxed_help
    assert "╭─ Target " in boxed_help
    assert "╭─ Network " in boxed_help
    assert "╭─ Auth " in boxed_help
    assert "╭─ 🌎 Global options " in boxed_help
    assert "╭─ 🤖 Agent options " in boxed_help
    assert "{rolling|canary}" in boxed_help
    assert "KEY=VAL" in boxed_help
    assert "INT..." in boxed_help
    assert "FLOAT FLOAT" in boxed_help

    # Every panel line in boxed_help must share the exact same terminal display width!
    panel_lines = [ln for ln in app.format_help().splitlines() if Colors.strip_ansi(ln).startswith(("╭", "│", "╰"))]
    assert len(panel_lines) > 15
    assert len({Colors.display_width(ln) for ln in panel_lines}) == 1

    # Grouped Markdown tables in `--help-md` (positional-only group, option-only group, mixed group)
    def mixed_group_fn(
        endpoint: Annotated[str, Arg(group="Connection")],
        timeout: Annotated[float, Arg(short="-t", group="Connection")] = 5.0,
    ) -> None:
        """Test mixed positional and option group in Markdown."""

    md_deploy = app.to_markdown()
    assert "**Positional Arguments**" in md_deploy
    assert "**Options**" in md_deploy
    assert "**Target**" in md_deploy
    assert "**Network**" in md_deploy
    assert "**Auth**" in md_deploy

    md_mixed = Cli(prog="mix", command=mixed_group_fn).to_markdown()
    assert "**Connection**" in md_mixed
    assert "| Parameter | Type / Choices | Default | Description |" in md_mixed

    # 4. Auto-degradation to unboxed help & unboxed parse error under CLIO_AGENT=1, TERM=dumb, or boxed=False
    monkeypatch.setenv("CLIO_AGENT", "1")
    agent_help = Colors.strip_ansi(app.format_help())
    assert "╭─" not in agent_help
    assert "Network:" in agent_help

    monkeypatch.delenv("CLIO_AGENT", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    dumb_help = Colors.strip_ansi(app.format_help())
    assert "╭─" not in dumb_help
    assert "Options:" in dumb_help

    monkeypatch.setenv("TERM", "xterm-256color")
    unboxed_app = Cli(
        prog="deployctl",
        command=deploy_cluster,
        formatter=HelpFormatter(boxed=False, show_types=False),
    )
    unboxed_help = Colors.strip_ansi(unboxed_app.format_help())
    assert "╭─" not in unboxed_help
    assert "{rolling|canary}" not in unboxed_help

    with pytest.raises(SystemExit) as err_exc:
        unboxed_app.run(["bundle.tar", "us-east", "--hos", "api.internal"])
    assert err_exc.value.code == 2
    unboxed_err = Colors.strip_ansi(capsys.readouterr().err)
    assert "╭─ Error" not in unboxed_err
    assert "Did you mean '--host'?" in unboxed_err
