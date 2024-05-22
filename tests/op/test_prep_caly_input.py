import os
import shutil
import unittest
from pathlib import (
    Path,
)

import numpy as np
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    TransientError,
)
from mock import (
    call,
    mock,
    patch,
)

# isort: off
from dpgen2.constants import (
    calypso_task_pattern,
    calypso_input_file,
    calypso_run_opt_file,
    calypso_check_opt_file,
)
from dpgen2.op.prep_caly_input import PrepCalyInput
from dpgen2.utils import (
    BinaryFileInput,
)

from dpgen2.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTask,
)
from dpgen2.exploration.task.calypso import (
    calypso_run_opt_str,
    calypso_run_opt_str_end,
    calypso_check_opt_str,
)

# isort: on


def make_task_group_list(njobs):
    tgrp = BaseExplorationTaskGroup()
    for ii in range(njobs):
        tt = ExplorationTask()
        tt.add_file(calypso_input_file, f"input.dat_{ii}")
        tt.add_file(
            calypso_run_opt_file,
            calypso_run_opt_str + calypso_run_opt_str_end % (0.01, 0.01),
        )
        tt.add_file(calypso_check_opt_file, calypso_check_opt_str)
        tgrp.add_task(tt)
    return tgrp


class TestPrepCalyInput(unittest.TestCase):
    def setUp(self):
        njobs = 2
        self.caly_task_grp = make_task_group_list(njobs)
        self.task_name_path = [calypso_task_pattern % i for i in range(2)]
        self.input_dat_list = [
            Path(i) / calypso_input_file for i in self.task_name_path
        ]
        self.caly_run_opt_list = [
            Path(i) / calypso_run_opt_file for i in self.task_name_path
        ]
        self.caly_check_opt_list = [
            Path(i) / calypso_check_opt_file for i in self.task_name_path
        ]

    def tearDown(self):
        for work_dir in self.task_name_path:
            shutil.rmtree(work_dir)

    def test_success(self):
        op = PrepCalyInput()
        out = op.execute(
            OPIO(
                {
                    "caly_task_grp": self.caly_task_grp,
                }
            )
        )
        # check output
        self.assertEqual(out["task_names"], self.task_name_path)
        self.assertEqual(out["input_dat_files"], self.input_dat_list)
        self.assertEqual(out["caly_run_opt_files"], self.caly_run_opt_list)
        self.assertEqual(out["caly_check_opt_files"], self.caly_check_opt_list)
        # check files details
        self.assertEqual(self.input_dat_list[0].read_text().strip("\n"), "input.dat_0")
        # self.assertEqual(self.caly_run_opt_list[1].read_text().strip("\n"), "run_1")
