import glob
import json
import os
import shutil
import sys
import textwrap
import unittest
from unittest.mock import patch, MagicMock
from pathlib import (
    Path,
)

import dpdata
import numpy as np

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.fp.cp2k import (
    Cp2kInputs,
    RunCp2k,
)

# isort: on


class TestCp2kInputs(unittest.TestCase):
    def setUp(self) -> None:
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe")

        self.data_path = Path(__file__).parent / "data.cp2k"

        with open(self.data_path / "input.yaml") as f:
            data = yaml.load(f)
        
        self.cp2k_input = Cp2kInputs(input_template=data)

    def test_make_cp2k_input(self):
        generated = self.cp2k_input.make_cp2k_input()
        with open(self.data_path / "input.inp") as f:
            expected = f.read()
        self.assertEqual(generated, expected)

    def test_make_cp2k_coord_cell(self):
        generated = self.cp2k_input.make_cp2k_coord_cell(
            sys_data=dpdata.System(self.data_path / "sys-2", fmt="deepmd/npy")
        )
        with open(self.data_path / "coord_n_cell.inc") as f:
            expected = f.read()
        self.assertEqual(generated, expected)

class TestRunCp2k(unittest.TestCase):
    @patch("dpgen2.fp.cp2k.run_command")
    @patch("dpgen2.fp.cp2k.dpdata.LabeledSystem")
    def test_run_task(self, mock_labeled_system, mock_run_command):
        # Mock the necessary objects and methods
        command = "cp2k"
        out = "output.data"
        log = "log.txt"
        ret = 0
        out_msg = "Output message"
        err_msg = "Error message"
        mock_run_command.return_value = (ret, out_msg, err_msg)
        mock_sys = MagicMock()
        mock_labeled_system.return_value = mock_sys

        # Create an instance of RunCp2k
        run_cp2k = RunCp2k()

        # Call the run_task method
        out_name, log_name = run_cp2k.run_task(command, out, log)

        # Assert the expected behavior
        mock_run_command.assert_called_once_with("cp2k > log.txt", shell=True)
        mock_labeled_system.assert_called_once_with(log, fmt="cp2k/output")
        mock_sys.to.assert_called_once_with("deepmd/npy", out)
        self.assertEqual(out_name, out)
        self.assertEqual(log_name, log)

    def test_input_files(self):
        # Create an instance of RunCp2k
        run_cp2k = RunCp2k()

        # Call the input_files method
        files = run_cp2k.input_files()

        # Assert the expected behavior
        self.assertEqual(files, ["coord_n_cell.inc", "input.inp"])
