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
from .context import (
    dpgen2,
)
from dpgen2.constants import (
    calypso_task_pattern,
    model_name_pattern,
    calypso_run_opt_file,
    calypso_check_opt_file,
)
from dpgen2.op import RunCalyDPOptim
from dpgen2.utils import (
    BinaryFileInput,
)

# isort: on


class TestRunDPOptim(unittest.TestCase):
    def setUp(self):
        self.task_name = "task_dir"
        self.task_dir = Path("pre_run_optim")

        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.joinpath("frozen_model.pb").write_text("1")
        self.task_dir.joinpath("calypso_run_opt.py").write_text("1")
        self.task_dir.joinpath("calypso_check_opt.py").write_text("1")
        for i in range(1, 11):
            self.task_dir.joinpath(f"POSCAR_{i}").write_text("1")

    def tearDown(self):
        shutil.rmtree(self.task_name)
        shutil.rmtree(self.task_dir)

    @patch("dpgen2.op.run_caly_dp_optim.run_command")
    def test_00_success(self, mocked_run):
        def side_effect(*args, **kwargs):
            for i in range(1, 11):
                Path().joinpath(f"CONTCAR_{str(i)}").write_text(f"CONTCAR_{str(i)}")
                Path().joinpath(f"OUTCAR_{str(i)}").write_text(f"OUTCAR_{str(i)}")
                Path().joinpath(f"{str(i)}.traj").write_text(f"{str(i)}.traj")
            return (0, "foo\n", "")

        mocked_run.side_effect = side_effect
        op = RunCalyDPOptim()
        out = op.execute(
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": self.task_name,
                    "finished": "false",
                    "cnt_num": 0,
                    "task_dir": self.task_dir,
                }
            )
        )
        # check output
        self.assertEqual(out["task_name"], self.task_name)

        optim_results_dir = out["optim_results_dir"]
        list_optim_results_dir = list(optim_results_dir.iterdir())
        counts_optim_results_dir = len(list_optim_results_dir)
        counts_outcar_in_optim_results_dir = len(
            list(optim_results_dir.rglob("OUTCAR_*"))
        )

        self.assertTrue(optim_results_dir, Path(self.task_name) / "optim_results_dir")
        self.assertEqual(counts_optim_results_dir, 30)
        self.assertEqual(counts_outcar_in_optim_results_dir, 10)
        self.assertTrue(
            Path(self.task_name) / "optim_results_dir" / "CONTCAR_4"
            in list_optim_results_dir
        )

        traj_results_dir = out["traj_results"]
        list_traj_results_dir = list(traj_results_dir.glob("*.traj"))
        counts_traj = len(list_traj_results_dir)
        self.assertEqual(traj_results_dir, Path(self.task_name) / "traj_results")
        self.assertEqual(counts_traj, 10)
        self.assertTrue(
            Path(self.task_name) / "traj_results" / "0.3.traj" in list_traj_results_dir
        )

    @patch("dpgen2.op.run_caly_dp_optim.run_command")
    def test_01_error(self, mocked_run):
        def side_effect(*args, **kwargs):
            for i in range(1, 6):
                Path().joinpath(f"CONTCAR_{str(i)}").write_text(f"CONTCAR_{str(i)}")
                Path().joinpath(f"OUTCAR_{str(i)}").write_text(f"OUTCAR_{str(i)}")
                Path().joinpath(f"{str(i)}.traj").write_text(f"{str(i)}.traj")
            return (1, "foo\n", "")

        mocked_run.side_effect = side_effect
        op = RunCalyDPOptim()
        self.assertRaises(
            TransientError,
            op.execute,
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": self.task_name,
                    "finished": "false",
                    "cnt_num": 0,
                    "task_dir": self.task_dir,
                }
            ),
        )

    def test_02_success(self):
        op = RunCalyDPOptim()
        out = op.execute(
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": self.task_name,
                    "finished": "true",
                    "cnt_num": 0,
                    "task_dir": self.task_dir,
                }
            )
        )
        # check output
        self.assertEqual(out["task_name"], self.task_name)

        optim_results_dir = out["optim_results_dir"]
        list_optim_results_dir = list(optim_results_dir.iterdir())
        counts_optim_results_dir = len(list_optim_results_dir)
        counts_outcar_in_optim_results_dir = len(
            list(optim_results_dir.rglob("OUTCAR_*"))
        )

        self.assertTrue(optim_results_dir, Path(self.task_name) / "optim_results_dir")
        self.assertEqual(counts_optim_results_dir, 0)
        self.assertEqual(counts_outcar_in_optim_results_dir, 0)
        self.assertTrue(
            Path(self.task_name) / "optim_results_dir" / "CONTCAR_4"
            not in list_optim_results_dir
        )

        traj_results_dir = out["traj_results"]
        list_traj_results_dir = list(traj_results_dir.glob("*.traj"))
        counts_traj = len(list_traj_results_dir)
        self.assertEqual(traj_results_dir, Path(self.task_name) / "traj_results")
        self.assertEqual(counts_traj, 0)
        self.assertTrue(
            Path(self.task_name) / "traj_results" / "0.3.traj"
            not in list_traj_results_dir
        )
