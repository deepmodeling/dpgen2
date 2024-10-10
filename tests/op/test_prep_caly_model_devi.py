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
from dpgen2.op import PrepCalyModelDevi
from dpgen2.utils import (
    BinaryFileInput,
)

# isort: on


class TestPrepCalyModelDevi(unittest.TestCase):
    def setUp(self):
        self.run_dir_name = "run_dir"

        self.ref_dir = Path("ref_dir")
        self.ref_dir.mkdir(parents=True, exist_ok=True)

        self.ref_traj_results = []

        ntrajs_dir = 5
        ntrajs_per_dir = 2
        for dir_index in range(ntrajs_dir):
            dir_name = self.ref_dir.joinpath(f"traj_dir_{dir_index}")
            dir_name.mkdir(parents=True, exist_ok=True)
            self.ref_traj_results.append(dir_name)
            for traj_index in range(ntrajs_per_dir):
                dir_name.joinpath(f"{dir_index}.{traj_index}.traj").write_text(
                    f"trajectory.{dir_index}.{traj_index}"
                )

        self.group_size = 5
        self.ngroup = ntrajs_dir * ntrajs_per_dir / self.group_size
        self.model_devi_group_size_2 = 0

    def tearDown(self):
        shutil.rmtree(self.ref_dir)
        shutil.rmtree(self.run_dir_name)

    def test_00_success(self):
        explore_config = {"model_devi_group_size": self.group_size}
        op = PrepCalyModelDevi()
        out = op.execute(
            OPIO(
                {
                    "task_name": self.run_dir_name,
                    "config": explore_config,
                    "traj_results": self.ref_traj_results,
                }
            )
        )
        # check output length
        self.assertEqual(len(out["task_name_list"]), self.ngroup)
        self.assertEqual(len(out["grouped_traj_list"]), self.ngroup)
        # check filename
        self.assertEqual(out["task_name_list"][0], "run_dir/trajs_part_0")
        self.assertEqual(out["task_name_list"][1], "run_dir/trajs_part_1")
        # check file exists
        self.assertTrue(Path(out["grouped_traj_list"][0]).exists())
        self.assertTrue(Path(out["grouped_traj_list"][1]).exists())

        traj_list = list(Path(out["grouped_traj_list"][0]).rglob("*traj"))
        # check traj number
        self.assertEqual(len(traj_list), 5)
        # check traj file name
        # self.assertTrue(Path("run_dir/trajs_part_0/0.0.0.traj") in traj_list)

    def test_01_success(self):
        explore_config = {}
        op = PrepCalyModelDevi()
        out = op.execute(
            OPIO(
                {
                    "task_name": self.run_dir_name,
                    "config": explore_config,
                    "traj_results": self.ref_traj_results,
                }
            )
        )
        # check output length
        self.assertEqual(len(out["task_name_list"]), 1)
        self.assertEqual(len(out["grouped_traj_list"]), 1)
        # check filename
        self.assertEqual(out["task_name_list"][0], "run_dir/trajs_part_0")
        # check file exists
        self.assertTrue(Path(out["grouped_traj_list"][0]).exists())

        traj_list = list(Path(out["grouped_traj_list"][0]).rglob("*traj"))
        # check traj number
        self.assertEqual(len(traj_list), 10)
        # check traj file name
        # self.assertTrue(Path("run_dir/trajs_part_0/0.0.0.traj") in traj_list)
