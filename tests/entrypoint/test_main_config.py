"""Exercise configuration loading at every CLI dispatch boundary."""

import importlib
import unittest
from unittest.mock import (
    patch,
)

from .context import (
    dpgen2,
)

main_module = importlib.import_module("dpgen2.entrypoint.main")


class TestMainConfig(unittest.TestCase):
    def test_every_configuration_command_uses_loader(self):
        cases = [
            ("submit", "submit_concurrent_learning"),
            ("resubmit", "resubmit_concurrent_learning"),
            ("status", "status"),
            ("showkey", "showkey"),
            ("download", "download_by_def"),
            ("watch", "watch"),
            ("restart", "submit_concurrent_learning"),
        ] + [
            (command, "execute_workflow_subcommand")
            for command in main_module.workflow_subcommands
        ]
        for command, entrypoint in cases:
            with self.subTest(command=command):
                argv = [command, "input.yaml"]
                if command != "submit":
                    argv.append("workflow-id")
                args = main_module.parse_args(argv)
                config = {"type_map": ["H", "O"]}
                with (
                    patch.object(main_module, "parse_args", return_value=args),
                    patch.object(
                        main_module, "load_config", return_value=config
                    ) as loader,
                    patch.object(main_module, entrypoint) as downstream,
                ):
                    main_module.main()
                    loader.assert_called_once_with("input.yaml")
                    downstream.assert_called_once()
                    self.assertTrue(
                        any(value is config for value in downstream.call_args.args)
                    )
                    if command == "restart":
                        downstream.return_value.submit.assert_called_once_with()
