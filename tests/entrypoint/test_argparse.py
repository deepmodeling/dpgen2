import json
import os
import shutil
import unittest

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.entrypoint.main import (
    main_parser,
    parse_args,
    workflow_subcommands,
)

# isort: off


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = main_parser()

    def test_commands(self):
        tested_commands = ["resubmit", "status", "download", "watch"]
        tested_commands += workflow_subcommands

        for cmd in tested_commands:
            parsed = self.parser.parse_args([cmd, "foo", "bar"])
            self.assertEqual(parsed.command, cmd)
            self.assertEqual(parsed.CONFIG, "foo")
            self.assertEqual(parsed.ID, "bar")

        tested_commands = ["submit"]
        for cmd in tested_commands:
            parsed = self.parser.parse_args([cmd, "foo"])
            self.assertEqual(parsed.command, cmd)
            self.assertEqual(parsed.CONFIG, "foo")

    def test_watch(self):
        parsed = self.parser.parse_args(
            [
                "watch",
                "foo",
                "bar",
                "-k",
                "foo",
                "bar",
                "tar",
                "-f",
                "10",
                "-d",
                "-p",
                "myprefix",
            ]
        )
        self.assertEqual(parsed.keys, ["foo", "bar", "tar"])
        self.assertEqual(parsed.download, True)
        self.assertEqual(parsed.frequency, 10)
        self.assertEqual(parsed.prefix, "myprefix")

    def test_dld(self):
        parsed = self.parser.parse_args(
            [
                "download",
                "foo",
                "bar",
                "-k",
                "foo",
                "bar",
                "tar",
                "-p",
                "myprefix",
            ]
        )
        self.assertEqual(parsed.keys, ["foo", "bar", "tar"])
        self.assertEqual(parsed.prefix, "myprefix")

    def test_download_by_definition(self):
        parsed = self.parser.parse_args(
            [
                "download",
                "input.json",
                "workflow-id",
                "--iterations",
                "0-2",
                "4",
                "--step-definitions",
                "prep-run-train/output/models",
                "prep-run-explore/output/trajs",
                "prep-run-fp/output/labeled_data",
                "--prefix",
                "results",
                "--no-check-point",
            ]
        )

        self.assertEqual(parsed.iterations, ["0-2", "4"])
        self.assertEqual(
            parsed.step_definitions,
            [
                "prep-run-train/output/models",
                "prep-run-explore/output/trajs",
                "prep-run-fp/output/labeled_data",
            ],
        )
        self.assertEqual(parsed.prefix, "results")
        self.assertFalse(parsed.no_check_point)

    def test_download_short_flags_and_defaults(self):
        parsed = self.parser.parse_args(
            [
                "download",
                "input.json",
                "workflow-id",
                "-i",
                "0-2",
                "-d",
                "collect-data/output/iter_data",
                "-n",
            ]
        )
        self.assertEqual(parsed.iterations, ["0-2"])
        self.assertEqual(parsed.step_definitions, ["collect-data/output/iter_data"])
        self.assertIsNone(parsed.keys)
        self.assertFalse(parsed.no_check_point)
        defaults = self.parser.parse_args(["download", "input.json", "workflow-id"])
        self.assertIsNone(defaults.keys)
        self.assertTrue(defaults.no_check_point)
        listed = self.parser.parse_args(["download", "input.json", "workflow-id", "-l"])
        self.assertTrue(listed.list_supported)

    def test_resubmit(self):
        parsed = self.parser.parse_args(
            [
                "resubmit",
                "foo",
                "bar",
                "-l",
                "--reuse",
                "0",
                "10-20",
            ]
        )
        self.assertEqual(parsed.list, True)
        self.assertEqual(parsed.reuse, ["0", "10-20"])
