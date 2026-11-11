# clio

**Lightweight, zero-dependency, agent-native CLI framework for Python 3.14+.**

`clio` pairs a native `@dataclass(frozen=True, slots=True)` token scanner (`ScanPlan`) and pre-compiled dispatch engine (`CallPlan`) with Python 3.14's `annotationlib` (PEP 649 / PEP 749), PEP 695 `type` aliases, pay-only-when-used `async` commands, agent-aware terminal spinners (`Spinner`), built-in JSON Schema / `SKILL.md` / stdio MCP support, and a zero-edit target runner (`uv run -m clio path/to/script.py[:my_fn]`) for wrapping arbitrary Python functions or modules without modifying source files.
