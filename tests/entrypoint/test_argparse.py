import unittest, json, shutil, os

from dpgen2.entrypoint.main import (
    main_parser,
    parse_args,
    workflow_subcommands,
)


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
