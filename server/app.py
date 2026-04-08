"""FastAPI entrypoint for InvoiceOps."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from invoiceops_env.models import InvoiceOpsAction, InvoiceOpsObservation
from invoiceops_env.server.invoiceops_env_environment import InvoiceOpsEnvironment


app = create_app(
    InvoiceOpsEnvironment,
    InvoiceOpsAction,
    InvoiceOpsObservation,
    env_name="invoiceops_env",
    max_concurrent_envs=4,
)


def _resolve_cli_args(
    default_host: str = "0.0.0.0",
    default_port: int = 8000,
) -> tuple[str, int]:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args, _ = parser.parse_known_args()
    return args.host, args.port


def main(host: str | None = None, port: int | None = None) -> None:
    """Run the server directly via ``uv run --project . server``."""
    import uvicorn

    if host is None and port is None:
        host, port = _resolve_cli_args()
    else:
        host = host or "0.0.0.0"
        port = port or 8000

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
