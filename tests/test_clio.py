r"""Comprehensive test suite for the cliop CLI library."""

import argparse
import io
import json
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
    HelpOnErrorParser,
    McpServer,
    PackageMeta,
    ParameterSpec,
    Style,
    ToolSpec,
)


# Fixtures & Helper Commands -------------------------------------------------------------------------------------------
class CustomDomainError(Exception):
    r"""Dummy domain exception for testing `catch_exceptions`."""


class LeafCommand(Command):
    r"""Execute a leaf action.

    Detailed multiline description for the leaf command.
    """

    aliases = ["lf"]

    def setup_arguments(self) -> None:
        assert self.parser is not None
        self.parser.add_argument("-n", "--name", default="world", help="Target name")

    def __call__(self, args: argparse.Namespace) -> None:
        assert self.cli is not None
        self.cli.msg(f"Running leaf for {args.name}")
        print(f"Hello, {args.name}!")


class ParentBranchCommand(CommandGroup):
    r"""Group of nested subcommands."""

    name = "branch"

    def get_shared_parser(self) -> argparse.ArgumentParser:
        shared = HelpOnErrorParser(add_help=False)
        shared.add_argument("--shared-flag", action="store_true", help="Shared branch flag")
        return shared

    def register_subcommands(self) -> None:
        self.subcommands.append(LeafCommand())


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

    # Shorthand style & color helpers
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

    # Semantic presets
    assert Colors.success("done") == f"{Colors.BOLD_GREEN}done{Colors.RESET}"
    assert Colors.warning("careful") == f"{Colors.BOLD_YELLOW}careful{Colors.RESET}"
    assert Colors.error("failed") == f"{Colors.BOLD_RED}failed{Colors.RESET}"
    assert Colors.info("note") == f"{Colors.BOLD_CYAN}note{Colors.RESET}"
    assert Colors.muted("subtle") == f"{Colors.DIM}subtle{Colors.RESET}"

    # Declarative style() builder
    composed = Colors.style("alert", fg=Colors.RED, bg=Colors.BG_WHITE, bold=True, underline=True)
    assert composed == f"{Colors.BOLD}{Colors.UNDERLINE}{Colors.RED}{Colors.BG_WHITE}alert{Colors.RESET}"

    # 256-color, RGB, and Hex builders
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

    # Strip & visible_len
    assert Colors.strip_ansi(composed) == "alert"
    assert Colors.visible_len(composed) == 5
    assert Colors.strip_ansi("no ansi") == "no ansi"

    # Frozen slotted Style dataclass & composition
    base_style = Style(fg=Colors.CYAN, bold=True)
    accent_style = base_style | Style(underline=True)
    assert accent_style("header") == f"{Colors.BOLD}{Colors.UNDERLINE}{Colors.CYAN}header{Colors.RESET}"
    with pytest.raises(AttributeError):
        base_style.bold = False  # type: ignore


# 2. Formatter & Parser Tests ------------------------------------------------------------------------------------------
def test_color_help_formatter_replaces_subcommands_and_injects_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    cli = Cli(prog="mytool", description="Top level tool", version="1.2.3")
    cli.add_command(LeafCommand())

    help_output = cli.parser.format_help()

    # Checks that usage has [options] and [subcommand] instead of {leafcommand,lf}
    assert "usage: mytool [options] [subcommand]" in help_output
    assert "{leafcommand" not in help_output
    assert "🌎 Global options:" in help_output
    assert "💬 Commands:" in help_output


def test_help_on_error_parser_fuzzy_match_suggestion(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Colors, "ENABLED", False)
    cli = Cli(prog="mytool")
    cli.add_command(LeafCommand())

    with pytest.raises(SystemExit) as exc_info:
        cli.run(["leafcomman"])  # Typo of 'leafcommand'

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage: mytool" in stderr
    assert "❌ Error: argument command: invalid choice: 'leafcomman'" in stderr
    assert "💡 Did you mean 'leafcommand'?" in stderr


# 3. Command Hierarchy & Execution Tests -------------------------------------------------------------------------------
def test_command_metadata_auto_population() -> None:
    cmd = LeafCommand()
    assert cmd.name == "leafcommand"
    assert cmd.help_text == "Execute a leaf action."
    assert "Detailed multiline description" in cmd.description


def test_nested_subcommands_and_shared_parser_inheritance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    received_args: dict[str, Any] = {}

    class InspectLeaf(Command):
        """Inspect inherited flags."""

        name = "inspect"

        def __call__(self, args: argparse.Namespace) -> None:
            received_args["verbose"] = args.verbose
            received_args["shared_flag"] = args.shared_flag

    class CustomBranch(ParentBranchCommand):
        def register_subcommands(self) -> None:
            self.subcommands.append(InspectLeaf())

    with Cli(prog="mytool") as app:
        app.add_command(CustomBranch())
        app.run(["branch", "inspect", "-V", "--shared-flag"])

    assert received_args["verbose"] is True
    assert received_args["shared_flag"] is True
    assert app.verbose is True


def test_branch_command_without_call_requires_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with Cli(prog="mytool") as app:
        app.add_command(ParentBranchCommand())
        with pytest.raises(SystemExit) as exc_info:
            app.run(["branch"])

    assert exc_info.value.code == 2
    assert "the following arguments are required: branch_subcommand" in capsys.readouterr().err


def test_version_flag_optional_registration(capsys: pytest.CaptureFixture[str]) -> None:
    # 1. With version provided
    cli_with_ver = Cli(prog="mytool", version="0.5.0")
    cli_with_ver.add_command(LeafCommand())
    with pytest.raises(SystemExit) as exc_info:
        cli_with_ver.run(["--version"])
    assert exc_info.value.code == 0
    assert "mytool 0.5.0" in capsys.readouterr().out

    # 2. Without version provided (-v should fail as unrecognized)
    cli_no_ver = Cli(prog="mytool")
    cli_no_ver.add_command(LeafCommand())
    with pytest.raises(SystemExit) as exc_info:
        cli_no_ver.run(["-v"])
    assert exc_info.value.code == 2


# 4. Error Trapping & Context Manager Signal Handling ------------------------------------------------------------------
def test_catch_cli_error_and_custom_exceptions(capsys: pytest.CaptureFixture[str]) -> None:
    class FailingCommand(Command):
        """Command that raises errors."""

        name = "fail"

        def setup_arguments(self) -> None:
            assert self.parser is not None
            self.parser.add_argument("--mode", choices=["cli", "domain"], default="cli")

        def __call__(self, args: argparse.Namespace) -> None:
            if args.mode == "cli":
                raise CliError("Custom CLI failure", exit_code=42)
            raise CustomDomainError("Database connection dropped")

    cli = Cli(prog="mytool", catch_exceptions=(CustomDomainError,))
    cli.add_command(FailingCommand())

    # Test built-in CliError with custom exit code
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["fail", "--mode", "cli"])
    assert exc_info.value.code == 42
    assert "❌ Custom CLI failure" in capsys.readouterr().err

    # Test custom domain exception via catch_exceptions
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
        # Ensure stdout/stdout.buffer are returned cleanly and never closed on exit
        stdout_handle = app.open_file("-", mode="w")
        stdout_bin_handle = app.open_file("stdout", mode="wb")
        assert stdout_handle is sys.stdout
        assert stdout_bin_handle is sys.stdout.buffer

    # Exiting context manager must close tracked file handle, but keep sys.stdout open
    assert handle.closed
    assert not sys.stdout.closed
    assert not sys.stdout.buffer.closed
    assert test_file.read_text() == "data"


def test_verbose_msg_and_progress_reporting(capsys: pytest.CaptureFixture[str]) -> None:
    cli = Cli(prog="mytool")

    # Silent when verbose=False
    cli.verbose = False
    cli.msg("Hidden log")
    items = list(cli.progress([1, 2, 3], "Processing"))
    assert items == [1, 2, 3]
    assert capsys.readouterr().err == ""

    # Emits to stderr when verbose=True (including generator without __len__)
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
    capsys: pytest.CaptureFixture[str],
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

    cli = Cli(prog="mytool")
    cli.add_command(cmd)

    assert cmd.parser is not None
    help_text = cmd.parser.format_help()
    assert "Path to input file." in help_text
    assert "Execution mode selector." in help_text
    assert "--no-cache" in help_text

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


def test_hybrid_setup_arguments_and_docstring_help_backfill(
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

        def setup_arguments(self) -> None:
            assert self.parser is not None
            # Explicitly register short flag -n/--name without a help string
            self.parser.add_argument("-n", "--name", default="default-name")

        def __call__(self, name: str = "default-name", retries: int = 3) -> None:
            captured["name"] = name
            captured["retries"] = retries

    cmd = HybridCommand()
    cli = Cli(prog="mytool")
    cli.add_command(cmd)

    assert cmd.parser is not None
    help_out = cmd.parser.format_help()
    # Help for -n/--name was back-filled from docstring, and --retries was auto-generated
    assert "-n, --name" in help_out
    assert "Documented name help from docstring." in help_out
    assert "--retries" in help_out
    assert "Number of retry attempts." in help_out

    cli.run(["hybrid", "-n", "custom", "--retries", "5"])
    assert captured == {"name": "custom", "retries": 5}


def test_add_arguments_from_external_class_and_callable() -> None:
    class ServerConfig:  # noqa: B903
        """Server configuration parameters.

        Args:
            host (str): Bind hostname.
            port (int): Listen port number.
        """

        def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
            self.host = host
            self.port = port

    captured: dict[str, Any] = {}

    class ServeCommand(Command):
        """Start server using external config class."""

        name = "serve"

        def setup_arguments(self) -> None:
            self.add_arguments_from(ServerConfig)

        def __call__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    cli = Cli(prog="mytool")
    cli.add_command(ServeCommand())
    cli.run(["serve", "--host", "0.0.0.0", "--port", "9090"])

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9090
    spec_obj = ParameterSpec(name="x", annotation=int, default=1)
    assert isinstance(spec_obj, ParameterSpec)
    with pytest.raises(AttributeError):
        spec_obj.name = "y"  # type: ignore


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
    assert isinstance(spec.docstring_spec, DocstringSpec)
    assert spec.call_plan.has_custom_call is True
    assert spec.call_plan.is_namespace_call is True

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

    # 1. Direct registration of callable @dataclass(frozen=True, slots=True)
    cli = Cli(prog="mltool", commands=[TrainJob, TuneCommand])
    cli.run(["train", "data/train.parquet", "--epochs", "20", "--tags", "v1", "fast"])

    assert len(executed) == 1
    job = executed[0]
    assert isinstance(job, TrainJob)
    assert job.dataset == Path("data/train.parquet")
    assert job.epochs == 20
    assert job.tags == ["v1", "fast"]

    # 2. Automatic @dataclass parameter expansion & instantiation in Command.__call__
    cli.run(["tune", "data/val.parquet", "--lr", "0.05", "--dry-run"])
    assert len(executed) == 2
    ds, opt, dry = executed[1]
    assert ds == Path("data/val.parquet")
    assert opt == OptimizerSpec(lr=0.05, weight_decay=0.01)
    assert dry is True

    # 3. Command.from_dataclass with an external handler function
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

    md = cli.to_markdown()
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

    # Standalone unbuilt command markdown generation
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

    # Non-existent package gracefully resolves to None
    assert PackageMeta.resolve("nonexistent-package-xyz-999") is None


def test_annotationlib_annotated_arg_metadata_and_forwardref_resilience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIO_TEST_TOKEN", "secret-token-42")
    monkeypatch.setenv("CLIO_TEST_WORKERS", "12")
    captured: dict[str, Any] = {}

    class ServeCommand(Command):
        """Start worker server.

        Args:
            workers: Docstring fallback description for workers.
        """

        name = "serve"

        def __call__(
            self,
            host: Annotated[str, Arg(positional=True, metavar="HOST"), "Bind hostname"] = "127.0.0.1",
            workers: Annotated[int, Arg(short="-w", metavar="N", env="CLIO_TEST_WORKERS")] = 4,
            token: Annotated[str, Arg(short="-t", env="CLIO_TEST_TOKEN", help="Auth token")] = "",
            mode: Annotated[str, Arg(short=("-m", "--server-mode"), choices=("fast", "safe"))] = "fast",
            context: UnresolvedFutureType | None = None,
        ) -> None:
            captured["host"] = host
            captured["workers"] = workers
            captured["token"] = token
            captured["mode"] = mode
            captured["context"] = context

    cli = Cli(prog="srv", commands=[ServeCommand])
    cli.run(["serve", "-m", "safe"])

    assert captured == {
        "host": "127.0.0.1",
        "workers": 12,
        "token": "secret-token-42",
        "mode": "safe",
        "context": None,
    }

    # Short flags override env defaults cleanly
    cli.run(["serve", "0.0.0.0", "-w", "32", "-t", "override"])
    assert captured["host"] == "0.0.0.0"
    assert captured["workers"] == 32
    assert captured["token"] == "override"

    md = cli.to_markdown()
    assert "`-w`, `--workers`" in md
    assert "`[env: CLIO_TEST_WORKERS]`" in md or "[env: CLIO_TEST_WORKERS]" in md
    assert "Docstring fallback description for workers." in md
    assert "Auth token" in md
    assert "Bind hostname" in md


# 7. Agentic Coding, Schema, SKILL.md, Dry-Run & MCP Tests -------------------------------------------------------------
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

    # 1. Typo in flag produces fuzzy suggestion and concise error (without full help dump)
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["work", "--coun", "5"])
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Did you mean '--count'?" in stderr
    assert "Run 'agenttool work --help' for full help." in stderr
    assert "🌎 Global options" not in stderr

    # 2. Progress in verbose agent mode emits a single summary line without \r frames
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

    # 1. --dry-run resolves CallPlan bindings without running BuildCommand.__call__
    dry_res = cli.run(["build", "dist/app.whl", "--replicas", "4", "--dry-run"])
    assert isinstance(dry_res, AgentResult)
    assert dry_res.ok is True
    assert dry_res.dry_run is True
    assert executed == []
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["dry_run"] is True
    assert dry_payload["arguments"] == {"target": "dist/app.whl", "replicas": 4, "fail": False}

    # 2. --json wraps return value and captures stdout + verbose cli.msg
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

    # 3. --json on CliError emits structured error envelope and exits with custom code
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["build", "dist/app.whl", "--fail", "--json"])
    assert exc_info.value.code == 9
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["ok"] is False
    assert err_payload["error"]["message"] == "Build failed cleanly"
    assert err_payload["error"]["exit_code"] == 9

    # 4. --schema and --skill flags exit 0 with structured output
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

    # 1. initialize, tools/list, tools/call via stdio stream
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

    assert len(responses) == 4  # Notification produces no response
    assert responses[0]["result"]["serverInfo"]["name"] == "calc"
    listed_names = [t["name"] for t in responses[1]["result"]["tools"]]
    assert listed_names == ["sum", "branch_leafcommand"]

    sum_res = responses[2]["result"]
    assert sum_res["isError"] is False
    assert sum_res["structuredContent"]["result"] == {"total": 42}

    leaf_res = responses[3]["result"]
    assert leaf_res["isError"] is False
    assert {"level": "stdout", "text": "Hello, agent!"} in leaf_res["structuredContent"]["messages"]


def test_extended_coverage_color_docstr_command_agent_and_cli(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import enum
    import inspect

    from clio.agent import serialize_for_json
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

    # 2. _parse_bool & serialize_for_json & ParameterSpec coercion / schema
    assert _parse_bool("yes") is True
    assert _parse_bool("off") is False
    with pytest.raises(argparse.ArgumentTypeError):
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

    p_empty = ParameterSpec(name="raw", annotation=argparse.Namespace)
    assert p_empty.type_label == "argparse.Namespace"
    assert ParameterSpec(name="untyped", annotation=inspect.Parameter.empty).type_label == "str"
    _, kwargs_pos_default = ParameterSpec(
        name="opt_pos", annotation=bool, default=True, arg_meta=Arg(positional=True)
    ).to_argparse()
    assert kwargs_pos_default["nargs"] == "?"

    # Coerce value & schema branches
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
    assert DocstringParser._unwrap_optional(int | None) == (int, True)
    DocstringParser.clear_cache()

    # 3. Imperative setup_arguments introspection, invoke_with_arguments, and CallPlan branches
    @dataclass
    class PosCfg:
        host: str = "localhost"

    class ImperativeCmd(Command):
        """Imperative command."""

        name = "imp"

        def setup_arguments(self) -> None:
            assert self.parser is not None
            self.parser.add_argument("target", help="Target name")
            self.parser.add_argument("-f", "--flag", action="store_true", help="Bool flag")
            self.parser.add_argument("--items", nargs="+", type=int, default=[1], help="Int items")
            self.parser.add_argument("--mode", choices=["fast", "slow"], default="fast", help="Mode")

        def __call__(
            self,
            ns: argparse.Namespace,
            cfg: PosCfg,
            /,
            *,
            extra_ns: argparse.Namespace,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if self.cli is not None:
                for _ in self.cli.progress([1, 2], "Iterating"):
                    pass
            print("err-line", file=sys.stderr)
            return {"host": cfg.host, "target": ns.target, "extra": extra_ns.mode, "kw": kwargs}

    imp_cmd = ImperativeCmd()
    specs = imp_cmd.get_parameter_specs()
    assert any(s.name == "target" for s in specs)
    assert imp_cmd.invoke_with_arguments({"target": "app", "flag": True})["target"] == "app"

    with pytest.raises(ValueError, match="cannot be null"):
        imp_cmd.invoke_with_arguments({"target": None})
    with pytest.raises(ValueError, match="Missing required argument"):
        imp_cmd.invoke_with_arguments({})
    with pytest.raises(TypeError, match="not executable"):
        ParentBranchCommand().invoke_with_arguments({})
    with pytest.raises(TypeError, match="Expected a BaseCommand"):
        Cli(prog="bad", commands=[123])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Expected a @dataclass"):
        Command.from_dataclass(int)
    with pytest.raises(TypeError, match="must either define '__call__'"):
        Command.from_dataclass(PosCfg)

    # 4. Custom Cli subclass, --json parser error, --help-md, --schema on subcommand, and MCP error branches
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

    # --help-md and subcommand --schema / --skill
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

    # MCP error handling branches (parse error, invalid request, ping, unknown method, invalid params, unknown tool)
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
