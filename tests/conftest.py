import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest CLI flags for benchmark documentation updates."""
    parser.addoption(
        "--update-benchmarks",
        action="store_true",
        default=False,
        help="Regenerate docs/benchmarks.md with latest benchmark measurements.",
    )


@pytest.fixture(scope="session")
def update_benchmarks(request: pytest.FixtureRequest) -> bool:
    """Return whether benchmark documentation should be overwritten on this run."""
    flag = bool(request.config.getoption("--update-benchmarks"))
    env = os.environ.get("CLIO_UPDATE_BENCHMARKS", "").strip().lower() in {"1", "true", "yes"}
    return flag or env
