import dataclasses
import enum
import io
import json
import os
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, TextIO

from clio.color import Colors

if TYPE_CHECKING:
    from clio.cli import Cli
    from clio.command import BaseCommand


def _detect_agent_mode(environ: Mapping[str, str] | None = None) -> bool:
    r"""Detect whether `CLIO_AGENT` environment mode is enabled.

    Args:
        environ (Mapping[str, str] | None): Environment dictionary to inspect.
            Defaults to `os.environ`.

    Returns:
        bool: `True` if `CLIO_AGENT` is set to a truthy value (`1`, `true`, `yes`, `on`).
    """
    env = os.environ if environ is None else environ
    raw = env.get("CLIO_AGENT", "").strip().lower()
    return raw in ("1", "true", "t", "yes", "y", "on")


def serialize_for_json(value: Any) -> Any:
    r"""Recursively convert arbitrary Python objects into JSON-serializable primitives.

    Handles `@dataclass` instances, `pathlib.Path` / `os.PathLike`, `enum.Enum`,
    sets, tuples, mappings, and objects implementing `.to_dict()`.

    Args:
        value (Any): Arbitrary Python value returned by a CLI command or default spec.

    Returns:
        Any: JSON-serializable structure (`dict`, `list`, `str`, `int`, `float`, `bool`, or `None`).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, enum.Enum):
        return serialize_for_json(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: serialize_for_json(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialize_for_json(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return serialize_for_json(value.to_dict())
    return str(value)


@dataclass(frozen=True, slots=True)
class AgentMessage:
    r"""Immutable diagnostic or output message captured during agent command execution.

    Attributes:
        level (str): Message channel or severity (`"info"`, `"success"`, `"warn"`,
            `"error"`, `"section"`, `"stdout"`, `"stderr"`).
        text (str): Cleaned text content of the message.
    """

    level: str
    text: str

    def to_dict(self) -> dict[str, str]:
        r"""Serialize message into a plain dictionary.

        Returns:
            dict[str, str]: Dictionary with `"level"` and `"text"` keys.
        """
        return {"level": self.level, "text": self.text}


@dataclass(frozen=True, slots=True)
class AgentErrorSpec:
    r"""Immutable structured error payload for agent and `--json` execution modes.

    Attributes:
        message (str): Human- and machine-readable error description.
        exit_code (int): Process exit status code associated with the error.
        usage (str | None): Compact CLI usage synopsis when triggered by a parser error.
        suggestions (tuple[str, ...]): Fuzzy-matched flag or choice suggestions, if any.
    """

    message: str
    exit_code: int = 1
    usage: str | None = None
    suggestions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        r"""Serialize error specification into a JSON-ready dictionary.

        Returns:
            dict[str, Any]: Structured error dictionary.
        """
        payload: dict[str, Any] = {
            "message": self.message,
            "exit_code": self.exit_code,
        }
        if self.usage:
            payload["usage"] = self.usage
        if self.suggestions:
            payload["suggestions"] = list(self.suggestions)
        return payload


@dataclass(frozen=True, slots=True)
class AgentResult:
    r"""Immutable execution envelope emitted by `--json`, `--dry-run`, and MCP `tools/call`.

    Attributes:
        ok (bool): Whether the command or validation succeeded.
        command (str | None): Resolved command path name (e.g. `"deploy"` or `"branch leafcommand"`).
        result (Any): Return value of the command callable (serialized for JSON).
        messages (tuple[AgentMessage, ...]): Captured log/stdout/stderr messages emitted during execution.
        error (AgentErrorSpec | None): Structured error details when `ok` is `False`.
        dry_run (bool): Whether execution was skipped via `--dry-run`.
        arguments (Mapping[str, Any] | None): Resolved `CallPlan` argument bindings when `dry_run` is `True`.
    """

    ok: bool
    command: str | None = None
    result: Any = None
    messages: tuple[AgentMessage, ...] = ()
    error: AgentErrorSpec | None = None
    dry_run: bool = False
    arguments: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        r"""Convert the execution envelope into a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Standardized envelope dictionary.
        """
        payload: dict[str, Any] = {
            "ok": self.ok,
            "command": self.command,
        }
        if self.dry_run:
            payload["dry_run"] = True
            payload["arguments"] = serialize_for_json(dict(self.arguments or {}))
        else:
            payload["result"] = serialize_for_json(self.result)
        payload["messages"] = [m.to_dict() for m in self.messages]
        payload["error"] = self.error.to_dict() if self.error is not None else None
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        r"""Serialize the execution envelope into a formatted JSON string.

        Args:
            indent (int | None): JSON indentation spaces (`None` for compact single-line JSON).

        Returns:
            str: JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    r"""Immutable LLM and Model Context Protocol (MCP) tool definition for a CLI command.

    Attributes:
        name (str): Sanitized tool identifier (e.g. `"deploy"` or `"branch_leafcommand"`).
        description (str): Cleaned prose description of the command.
        input_schema (dict[str, Any]): JSON Schema (`type: "object"`) describing command parameters.
        command_path (tuple[str, ...]): Sequence of CLI command names to reach this leaf command.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    command_path: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        r"""Return the MCP / standard LLM tool dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing `"name"`, `"description"`, and `"inputSchema"`.
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        r"""Return an OpenAI function-calling compatible tool definition dictionary.

        Returns:
            dict[str, Any]: Dictionary formatted as `{"type": "function", "function": {...}}`.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class _OutputCaptureContext:
    r"""Internal context manager that intercepts `stdout`, `stderr`, and `Cli` messages during `--json` or MCP runs."""

    def __init__(self, cli: Cli) -> None:
        self.cli = cli
        self.messages: list[AgentMessage] = []
        self._stdout_buf = io.StringIO()
        self._stderr_buf = io.StringIO()
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None
        self._prev_capture: list[AgentMessage] | None = None

    def __enter__(self) -> Self:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._prev_capture = self.cli._active_message_sink
        self.cli._active_message_sink = self.messages
        sys.stdout = self._stdout_buf
        sys.stderr = self._stderr_buf
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        self.cli._active_message_sink = self._prev_capture

        out_text = Colors.strip_ansi(self._stdout_buf.getvalue()).strip()
        if out_text:
            for line in out_text.splitlines():
                self.messages.append(AgentMessage(level="stdout", text=line))

        err_text = Colors.strip_ansi(self._stderr_buf.getvalue()).strip()
        if err_text:
            for line in err_text.splitlines():
                self.messages.append(AgentMessage(level="stderr", text=line))


class McpServer:
    r"""Zero-dependency Model Context Protocol (MCP) stdio JSON-RPC 2.0 server for a [`Cli`][clio.cli.Cli] app.

    Exposes all executable commands registered on a `Cli` instance as MCP tools over
    standard input/output using only the Python standard library.

    Attributes:
        cli (Cli): Host `Cli` application whose commands are exposed as MCP tools.
        protocol_version (str): Negotiated MCP protocol version string.
    """

    def __init__(self, cli: Cli, *, protocol_version: str = "2024-11-05") -> None:
        r"""Initialize an MCP server bound to a `Cli` instance.

        Args:
            cli (Cli): Target `Cli` instance.
            protocol_version (str): MCP protocol version string. Defaults to `"2024-11-05"`.
        """
        self.cli = cli
        self.protocol_version = protocol_version
        self._tool_map: dict[str, tuple[ToolSpec, BaseCommand]] = {}
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        r"""Build lookup table mapping tool names to `(ToolSpec, BaseCommand)` pairs."""
        self._tool_map.clear()
        for cmd in self.cli.commands:
            for tool_spec, leaf_cmd in cmd._iter_tool_bindings():
                self._tool_map[tool_spec.name] = (tool_spec, leaf_cmd)

    def list_tools(self) -> list[dict[str, Any]]:
        r"""Return MCP tool definitions for all executable commands in the CLI.

        Returns:
            list[dict[str, Any]]: List of MCP tool dictionaries (`name`, `description`, `inputSchema`).
        """
        self._refresh_tools()
        return [spec.to_dict() for spec, _ in self._tool_map.values()]

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> AgentResult:
        r"""Invoke a registered CLI command by its MCP tool name with structured arguments.

        Args:
            name (str): Tool name as advertised by `list_tools()`.
            arguments (Mapping[str, Any] | None): Dictionary of tool arguments keyed by parameter name.

        Returns:
            AgentResult: Structured execution result envelope.
        """
        self._refresh_tools()
        binding = self._tool_map.get(name)
        if binding is None:
            return AgentResult(
                ok=False,
                command=name,
                error=AgentErrorSpec(
                    message=f"Unknown tool: '{name}'",
                    exit_code=2,
                    suggestions=tuple(self._tool_map.keys()),
                ),
            )

        tool_spec, leaf_cmd = binding
        cmd_label = " ".join(tool_spec.command_path) if tool_spec.command_path else leaf_cmd.name
        with _OutputCaptureContext(self.cli) as cap:
            try:
                ret = leaf_cmd.invoke_with_arguments(arguments or {})
            except (*self.cli.catch_exceptions, ValueError, TypeError) as exc:
                exit_code = getattr(exc, "exit_code", 1)
                return AgentResult(
                    ok=False,
                    command=cmd_label,
                    messages=tuple(cap.messages),
                    error=AgentErrorSpec(message=str(exc), exit_code=exit_code),
                )

        return AgentResult(
            ok=True,
            command=cmd_label,
            result=ret,
            messages=tuple(cap.messages),
        )

    def handle_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        r"""Process a single decoded JSON-RPC 2.0 message and return a response dictionary.

        Args:
            message (Mapping[str, Any]): Parsed JSON-RPC 2.0 request or notification object.

        Returns:
            dict[str, Any] | None: JSON-RPC 2.0 response object, or `None` for notifications.
        """
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # JSON-RPC notifications do not carry an "id" and expect no response
        if "id" not in message:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": self.cli.parser.prog or "clio",
                        "version": self.cli.config.version or "0.1.0",
                    },
                },
            }

        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {},
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": self.list_tools(),
                },
            }

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments")
            if not isinstance(tool_name, str) or not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: missing tool 'name'.",
                    },
                }
            if tool_args is not None and not isinstance(tool_args, Mapping):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'arguments' must be an object.",
                    },
                }
            agent_result = self.call_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": agent_result.to_json()}],
                    "structuredContent": agent_result.to_dict(),
                    "isError": not agent_result.ok,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    def serve(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        r"""Run the blocking stdio JSON-RPC 2.0 loop until EOF on `stdin`.

        Args:
            stdin (TextIO | None): Input stream for JSON-RPC lines. Defaults to `sys.stdin`.
            stdout (TextIO | None): Output stream for JSON-RPC responses. Defaults to `sys.stdout`.
        """
        in_stream = sys.stdin if stdin is None else stdin
        out_stream = sys.stdout if stdout is None else stdout

        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                out_stream.write(json.dumps(err_resp) + "\n")
                out_stream.flush()
                continue

            if not isinstance(payload, Mapping):
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: expected JSON object."},
                }
                out_stream.write(json.dumps(err_resp) + "\n")
                out_stream.flush()
                continue

            response = self.handle_message(payload)
            if response is not None:
                out_stream.write(json.dumps(response) + "\n")
                out_stream.flush()
