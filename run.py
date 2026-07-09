"""Command-line entrypoint for the WebLLM2API server."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence

import uvicorn

from app_logging import LOG_LEVELS, build_logging_config, configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.getenv("PORT", "8000")), type=int)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=tuple(LOG_LEVELS),
        default="info",
        help="console verbosity (default: info)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    numeric_level = configure_logging(args.log_level)
    logger = logging.getLogger("webllm2api")
    logger.info(
        "Starting WebLLM2API on http://%s:%s (log level: %s)",
        args.host,
        args.port,
        args.log_level,
    )

    if args.reload:
        application = "openai_server:app"
    else:
        from openai_server import app

        application = app

    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=build_logging_config(args.log_level),
        log_level=numeric_level,
    )


if __name__ == "__main__":
    main()
