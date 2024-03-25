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
from dpgen2.op.prep_run_dp_optim import PrepRunDPOptim
from dpgen2.utils import (
    BinaryFileInput,
)

# isort: on


class TestPrepRunDPOptim(unittest.TestCase):
    def setUp(self):
        self.config_1 = {"run_calypso_command": "echo 1"}
        self.config_2 = {"run_calypso_command": None}

        self.poscar_dir = Path("poscar_dir")
        self.poscar_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 6):
            self.poscar_dir.joinpath(f"POSCAR_{str(i)}").write_text(f"POSCAR_{str(i)}")

        self.models_dir = Path("models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            path_name = self.models_dir.joinpath(f"task.{i}")
            path_name.mkdir(exist_ok=True, parents=True)
            path_name.joinpath("frozen_model.pb").write_text("frozen_model.pb")

        self.task_name = calypso_task_pattern % 0

        self.other_files = Path("other_files")
        self.other_files.mkdir(parents=True, exist_ok=True)
        self.caly_run_opt_file = self.other_files.joinpath(calypso_run_opt_file)
        self.caly_run_opt_file.write_text("run_opt_file")
        self.caly_check_opt_file = self.other_files.joinpath(calypso_check_opt_file)
        self.caly_check_opt_file.write_text("check_opt_file")

        self.ref_optim_results_dir = Path("optim_results_dir")
        self.ref_optim_results_dir.mkdir(parents=True, exist_ok=True)
        for temp in self.poscar_dir.iterdir():
            self.ref_optim_results_dir.joinpath(temp.name).symlink_to(temp.resolve())
            self.ref_optim_results_dir.joinpath(
                str(temp.name).replace("POSCAR", "CONTCAR")
            ).write_text("foo")
            self.ref_optim_results_dir.joinpath(
                str(temp.name).replace("POSCAR", "OUTCAR")
            ).write_text("foo")

        self.ref_traj_results_dir = Path("traj_results_dir")
        self.ref_traj_results_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, len(list(self.poscar_dir.iterdir())) + 1):
            self.ref_optim_results_dir.joinpath(f"{str(i+1)}.traj").write_text("foo")

    def tearDown(self):
        shutil.rmtree(self.other_files)
        shutil.rmtree(self.ref_optim_results_dir)
        shutil.rmtree(self.ref_traj_results_dir)
        shutil.rmtree(Path(self.task_name))
        shutil.rmtree(self.poscar_dir)
        shutil.rmtree(self.models_dir)

    @patch("dpgen2.op.prep_run_dp_optim.run_command")
    def test_success_00(self, mocked_run):
        if Path(self.task_name).is_dir():
            shutil.rmtree(Path(self.task_name))

        def side_effect(*args, **kwargs):
            for i in range(1, 6):
                Path().joinpath(f"CONTCAR_{str(i)}").write_text(f"CONTCAR_{str(i)}")
                Path().joinpath(f"OUTCAR_{str(i)}").write_text(f"OUTCAR_{str(i)}")
                Path().joinpath(f"{str(i)}.traj").write_text(f"{str(i)}.traj")
            return (0, "foo\n", "")

        mocked_run.side_effect = side_effect
        op = PrepRunDPOptim()
        out = op.execute(
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": calypso_task_pattern % 0,
                    "finished": "false",
                    "cnt_num": 0,
                    "poscar_dir": self.poscar_dir,
                    "models_dir": self.models_dir,
                    "caly_run_opt_file": self.caly_run_opt_file,
                    "caly_check_opt_file": self.caly_check_opt_file,
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
        self.assertEqual(counts_optim_results_dir, 15)
        self.assertEqual(counts_outcar_in_optim_results_dir, 5)
        self.assertTrue(
            Path(self.task_name) / "optim_results_dir" / "CONTCAR_4"
            in list_optim_results_dir
        )

        traj_results_dir = out["traj_results"]
        list_traj_results_dir = list(traj_results_dir.glob("*.traj"))
        counts_traj = len(list_traj_results_dir)
        self.assertEqual(traj_results_dir, Path(self.task_name) / "traj_results")
        self.assertEqual(counts_traj, 5)
        self.assertTrue(
            Path(self.task_name) / "traj_results" / "0.3.traj" in list_traj_results_dir
        )

        self.assertEqual(
            Path(self.task_name) / calypso_run_opt_file, out["caly_run_opt_file"]
        )
        self.assertEqual(
            Path(self.task_name) / calypso_check_opt_file, out["caly_check_opt_file"]
        )

    @patch("dpgen2.op.prep_run_dp_optim.run_command")
    def test_error_01(self, mocked_run):
        if Path(self.task_name).is_dir():
            shutil.rmtree(Path(self.task_name))

        def side_effect(*args, **kwargs):
            for i in range(1, 6):
                Path().joinpath(f"CONTCAR_{str(i)}").write_text(f"CONTCAR_{str(i)}")
                Path().joinpath(f"OUTCAR_{str(i)}").write_text(f"OUTCAR_{str(i)}")
                Path().joinpath(f"{str(i)}.traj").write_text(f"{str(i)}.traj")
            return (1, "foo\n", "")

        mocked_run.side_effect = side_effect
        op = PrepRunDPOptim()
        self.assertRaises(
            TransientError,
            op.execute,
            OPIO(
                {
                    "config": {"run_calypso_command": "echo 1"},
                    "task_name": calypso_task_pattern % 0,
                    "finished": "false",
                    "cnt_num": 0,
                    "poscar_dir": self.poscar_dir,
                    "models_dir": self.models_dir,
                    "caly_run_opt_file": self.caly_run_opt_file,
                    "caly_check_opt_file": self.caly_check_opt_file,
                }
            ),
        )
