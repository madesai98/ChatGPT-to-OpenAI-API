from __future__ import annotations

import logging
import unittest

from app_logging import (
    LOG_LEVELS,
    VERBOSE,
    build_logging_config,
    configure_logging,
)
from run import build_parser


class LoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_logging("info")

    def test_default_log_level_is_info(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.log_level, "info")

    def test_all_five_log_levels_are_available(self) -> None:
        self.assertEqual(
            tuple(LOG_LEVELS),
            ("verbose", "debug", "info", "warning", "error"),
        )
        self.assertEqual(LOG_LEVELS["verbose"], VERBOSE)

    def test_configure_logging_sets_root_threshold(self) -> None:
        configure_logging("verbose")
        self.assertEqual(logging.getLogger().level, VERBOSE)

    def test_uvicorn_config_uses_selected_threshold(self) -> None:
        config = build_logging_config("warning")
        self.assertEqual(config["root"]["level"], logging.WARNING)

    def test_cli_accepts_each_log_level(self) -> None:
        parser = build_parser()
        for level in LOG_LEVELS:
            with self.subTest(level=level):
                self.assertEqual(
                    parser.parse_args(["--log-level", level]).log_level,
                    level,
                )


if __name__ == "__main__":
    unittest.main()
