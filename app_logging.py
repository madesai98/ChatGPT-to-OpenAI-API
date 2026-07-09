"""Shared console logging configuration for WebLLM2API."""

from __future__ import annotations

import logging
import logging.config
from typing import Final


VERBOSE: Final = 5
LOG_LEVELS: Final[dict[str, int]] = {
    "verbose": VERBOSE,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

logging.addLevelName(VERBOSE, "VERBOSE")


def resolve_log_level(level: str) -> int:
    normalized = level.lower()
    if normalized not in LOG_LEVELS:
        choices = ", ".join(LOG_LEVELS)
        raise ValueError(f"Unknown log level {level!r}; choose from: {choices}")
    return LOG_LEVELS[normalized]


def build_logging_config(level: str = "info") -> dict[str, object]:
    """Build a config Uvicorn can also apply inside reload subprocesses."""
    numeric_level = resolve_log_level(level)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            }
        },
        "root": {
            "handlers": ["default"],
            "level": numeric_level,
        },
        "loggers": {
            "uvicorn": {
                "handlers": [],
                "level": numeric_level,
                "propagate": True,
            },
            "uvicorn.error": {
                "level": numeric_level,
                "propagate": True,
            },
            "uvicorn.access": {
                "level": numeric_level,
                "propagate": True,
            },
        },
    }


def configure_logging(level: str = "info") -> int:
    """Configure application and Uvicorn logs on one five-level scale."""
    numeric_level = resolve_log_level(level)
    logging.config.dictConfig(build_logging_config(level))
    logging.captureWarnings(True)
    return numeric_level
