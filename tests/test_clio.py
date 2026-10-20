r"""Comprehensive test suite for the cliop CLI library."""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pytest

if TYPE_CHECKING:

    class UnresolvedFutureType:
        """Type defined only during static type checking to test runtime PEP 649 ForwardRef handling."""


from clio import (
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
    PackageMeta,
    ParameterSpec,
    Style,
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
