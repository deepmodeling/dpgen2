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
from dpgen2.op import PrepCalyDPOptim
from dpgen2.utils import (
    BinaryFileInput,
)

# isort: on


class TestPrepDPOptim(unittest.TestCase):
    def setUp(self):
        self.config_1 = {"run_calypso_command": "echo 1"}
        self.config_2 = {"run_calypso_command": None}

        self.poscar_dir = Path("poscar_dir")
        self.poscar_dir.mkdir(parents=True, exist_ok=True)
        nposcar = 10
        for i in range(1, nposcar + 1):
            self.poscar_dir.joinpath(f"POSCAR_{str(i)}").write_text(f"POSCAR_{str(i)}")

        self.models_dir = Path("models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            model_path = self.models_dir.joinpath(f"task_name.{i}")
            model_path.mkdir(parents=True, exist_ok=True)
            model_path.joinpath("frozen_model.pb").write_text("pb")

        self.task_name = calypso_task_pattern % 0

        self.other_files = Path("other_files")
        self.other_files.mkdir(parents=True, exist_ok=True)
        self.caly_run_opt_file = self.other_files.joinpath(calypso_run_opt_file)
        self.caly_run_opt_file.write_text("run_opt_file")
        self.caly_check_opt_file = self.other_files.joinpath(calypso_check_opt_file)
        self.caly_check_opt_file.write_text("check_opt_file")

        self.template_slice_config = {"group_size": 3}
        self.group_size = self.template_slice_config["group_size"]

        grouped_poscar_list = [i for i in range(0, nposcar, self.group_size)]
        self.ngrouped = len(grouped_poscar_list)
        self.ref_task_dirs = []
        for i in range(0, self.ngrouped):
            self.ref_task_dirs += [self.task_name / Path(f"opt_path_{i}")]
        self.ref_task_names = [str(task_dir) for task_dir in self.ref_task_dirs]

    def tearDown(self):
        shutil.rmtree(self.other_files)
        shutil.rmtree(self.task_name)
        shutil.rmtree(self.poscar_dir)

    def test_success_00(self):
        if Path(self.task_name).is_dir():
            shutil.rmtree(Path(self.task_name))

        op = PrepCalyDPOptim()
        out = op.execute(
            OPIO(
                {
                    "task_name": calypso_task_pattern % 0,
                    "finished": "false",
                    "template_slice_config": self.template_slice_config,
                    "poscar_dir": self.poscar_dir,
                    "models_dir": self.models_dir,
                    "caly_run_opt_file": self.caly_run_opt_file,
                    "caly_check_opt_file": self.caly_check_opt_file,
                }
            )
        )
        # check output
        self.assertEqual(len(out["task_names"]), self.ngrouped)
        self.assertEqual(len(out["task_dirs"]), self.ngrouped)
        self.assertEqual(out["task_names"], self.ref_task_names)

        first_opt_path, last_opt_path = out["task_dirs"][0], out["task_dirs"][-1]

        ref_common_opt_path_file_check = [
            "frozen_model.pb",
            "calypso_run_opt.py",
            "calypso_check_opt.py",
        ]
        ref_first_opt_path_file_check = [
            f"POSCAR_{i}" for i in range(1, self.ngrouped + 1)
        ]
        first_opt_path_files = sorted(Path(first_opt_path).iterdir())
        for file_name in first_opt_path_files:
            self.assertTrue(
                file_name.name
                in ref_first_opt_path_file_check + ref_common_opt_path_file_check
            )

        ref_last_opt_path_file_check = ["POSCAR_10"]
        last_opt_path_files = sorted(last_opt_path.iterdir())
        for file_name in last_opt_path_files:
            self.assertTrue(
                file_name.name
                in ref_last_opt_path_file_check + ref_common_opt_path_file_check
            )

    def test_01_success(self):
        if Path(self.task_name).is_dir():
            shutil.rmtree(Path(self.task_name))

        op = PrepCalyDPOptim()
        out = op.execute(
            OPIO(
                {
                    "task_name": calypso_task_pattern % 0,
                    "finished": "true",
                    "template_slice_config": self.template_slice_config,
                    "poscar_dir": self.poscar_dir,
                    "models_dir": self.models_dir,
                    "caly_run_opt_file": self.caly_run_opt_file,
                    "caly_check_opt_file": self.caly_check_opt_file,
                }
            )
        )
        # check output
        self.assertEqual(len(out["task_names"]), 1)
        self.assertEqual(len(out["task_dirs"]), 1)
        self.assertEqual(
            out["task_names"], [str(Path(self.task_name) / Path("opt_path"))]
        )
