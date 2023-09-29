import json
import os
import shutil
import unittest
from pathlib import (
    Path,
)

import numpy as np

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.utils.run_command import (
    run_command,
)

# isort: on


class TestRunCommand(unittest.TestCase):
    def setUp(self):
        self.work_path = Path("work_path")
        self.work_path.mkdir(exist_ok=True)
        (self.work_path / "foo").write_text("foo")
        (self.work_path / "bar").write_text("foo")

    def tearDown(self):
        if self.work_path.is_dir():
            shutil.rmtree(self.work_path)

    def test_success_shell(self):
        os.chdir(self.work_path)
        ret, out, err = run_command(["ls | sort"], shell=True)
        self.assertEqual(ret, 0)
        self.assertEqual(out, "bar\nfoo\n")
        # ignore the warnings
        # self.assertEqual(err, '')
        os.chdir("..")

    def test_success(self):
        os.chdir(self.work_path)
        ret, out, err = run_command(["ls"])
        self.assertEqual(ret, 0)
        self.assertEqual(out, "bar\nfoo\n")
        self.assertEqual(err, "")
        os.chdir("..")

    def test_success_foo(self):
        os.chdir(self.work_path)
        ret, out, err = run_command(["ls", "foo"])
        self.assertEqual(ret, 0)
        self.assertEqual(out, "foo\n")
        self.assertEqual(err, "")
        os.chdir("..")

    def test_failed(self):
        os.chdir(self.work_path)
        ret, out, err = run_command(["ls", "tar"])
        self.assertNotEqual(ret, 0)
        self.assertEqual(out, "")
        # self.assertEqual(err, "ls: cannot access 'tar': No such file or directory\n")
        self.assertNotEqual(err, "")
        os.chdir("..")
