import os
import shutil
import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    Mock,
    call,
    patch,
)

import numpy as np
from ase import (
    Atoms,
)
from ase.io import (
    write,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    calypso_input_file,
    calypso_log_name,
    calypso_task_pattern,
)
from dpgen2.op.run_caly_model_devi import (
    RunCalyModelDevi,
    atoms2lmpdump,
    parse_traj,
)
from dpgen2.utils import (
    BinaryFileInput,
)

# from .context import (
#     dpgen2,
# )

# isort: on
try:
    import deepmd

    x = 0
except:
    x = 1


class TestRunCalyModelDevi(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path().joinpath("caly_model_devi")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.atoms_normal_2 = Atoms(
            numbers=[1, 2],
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )
        self.atoms_normal_1 = Atoms(
            numbers=[1],
            scaled_positions=[[0, 0, 0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )
        self.atoms_normal_3 = Atoms(
            numbers=[1, 2, 3],
            scaled_positions=[
                [0, 0, 0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
            ],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )

        self.atoms_normal_4 = Atoms(
            numbers=[1, 2, 3],
            scaled_positions=[
                [0, 0, 0],
                [0.0, 0.0, 0.25],
                [0.5, 0.0, 0.6],
            ],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )
        self.atoms_normal_5 = Atoms(
            numbers=[1, 2, 3],
            scaled_positions=[
                [0, 0, 0],
                [0.0, 0.0, 0.3],
                [0.5, 0.0, 0.6],
            ],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )
        self.atoms_normal_6 = Atoms(
            numbers=[1, 2, 3],
            scaled_positions=[
                [0, 0, 0],
                [0.0, 0.0, 0.2],
                [0.5, 0.0, 0.7],
            ],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )
        sa6 = self.atoms_normal_6
        self.atoms_abnormal_2 = Atoms(
            numbers=[1, 2, 3, 3],
            scaled_positions=[
                [0, 0, 0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.0, 0.6],
            ],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )
        self.atoms_abnormal = Atoms(
            numbers=[1, 2],
            scaled_positions=[[0.00001, 0, 0], [0.00002, 0.0, 0.0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )

        self.atoms_abnormalpbc = Atoms(
            numbers=[1, 2],
            scaled_positions=[[0.00001, 0, 0], [0.99999, 0.0, 0.0]],
            cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )
        self.traj_file_1 = self.work_dir.joinpath("1.traj")
        self.traj_file_2 = self.work_dir.joinpath("2.traj")
        self.traj_file_3 = self.work_dir.joinpath("3.traj")
        self.traj_file_4 = self.work_dir.joinpath("4.traj")
        self.traj_file_5 = self.work_dir.joinpath("5.traj")
        self.traj_file_6 = self.work_dir.joinpath("6.traj")
        self.traj_file_10 = self.work_dir.joinpath(
            "10.traj"
        )  # in traj 10, test the abnormal configuration.
        self.traj_file_20 = self.work_dir.joinpath(
            "20.traj"
        )  # in traj 20, test the mix of normal and abnormal configuration.
        self.traj_file_21 = self.work_dir.joinpath(
            "21.traj"
        )  # in traj 21, test the mix of normal and abnormal configuration across pbc.

        write(
            self.traj_file_1,
            [self.atoms_normal_1],
            format="traj",
        )

        write(
            self.traj_file_2,
            [self.atoms_normal_2],
            format="traj",
        )

        write(
            self.traj_file_3,
            [self.atoms_normal_3, self.atoms_abnormal_2],
            format="traj",
        )

        write(
            self.traj_file_4,
            [
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                sa6,
                self.atoms_abnormal_2,
            ],
            format="traj",
        )

        write(
            self.traj_file_5,
            [self.atoms_normal_3],
            format="traj",
        )

        write(
            self.traj_file_6,
            [self.atoms_normal_3],
            format="traj",
        )
        ### The abnormal case ###
        write(self.traj_file_10, self.atoms_abnormalpbc, format="traj")

        ### The mixed case ###
        write(
            self.traj_file_20, [self.atoms_normal_3, self.atoms_abnormal], format="traj"
        )
        write(
            self.traj_file_21,
            [self.atoms_normal_1, self.atoms_abnormalpbc],
            format="traj",
        )

        self.ref_dump_str = """ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
        0.0000000000        10.0000000000         0.0000000000
        0.0000000000        10.0000000000         0.0000000000
        0.0000000000        10.0000000000         0.0000000000
ITEM: ATOMS id type x y z fx fy fz
    1     1        0.0000000000         0.0000000000         0.0000000000        0.0000000000         0.0000000000         0.0000000000
    2     2        5.0000000000         5.0000000000         5.0000000000        0.0000000000         0.0000000000         0.0000000000
"""
        self.type_map = ["H", "He", "Li", "Na"]
        self.task_name = self.work_dir.joinpath(calypso_task_pattern % 0)
        self.traj_dirs = [
            self.traj_file_1,
            self.traj_file_2,
            self.traj_file_3,
            self.traj_file_4,
            self.traj_file_5,
            self.traj_file_6,
            self.traj_file_10,
        ]

        self.model_1 = self.work_dir.joinpath("model.000.pb")
        self.model_2 = self.work_dir.joinpath("model.001.pb")
        self.model_1.write_text("model.000.pb")
        self.model_2.write_text("model.001.pb")
        self.models = [self.model_1, self.model_2]

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_00_parse_traj(self):
        atoms_list_3 = parse_traj(self.traj_file_3)
        self.assertEqual(len(atoms_list_3), 1)  # reasonable results are added

        atoms_list_4 = parse_traj(self.traj_file_4)
        self.assertEqual(len(atoms_list_4), 4)  # reasonable results are added

        atoms_list_20 = parse_traj(self.traj_file_20)
        atoms_list_21 = parse_traj(self.traj_file_21)

        self.assertEqual(len(atoms_list_20), 1)  # unreasonable results are omitted
        self.assertEqual(
            len(atoms_list_21), 1
        )  # unreasonable results are omitted, if 2, ase ignors dangerous distances across pbc

        self.assertAlmostEqual(atoms_list_3[-1], self.atoms_normal_3)
        atoms_list_10 = parse_traj(self.traj_file_10)
        self.assertTrue(atoms_list_10 is None), self.atoms_abnormal

    def test_01_atoms2lmpdump(self):
        dump_str = atoms2lmpdump(self.atoms_normal_2, 1, self.type_map)
        self.assertEqual(dump_str, self.ref_dump_str)

    # @patch("dpgen2.op.run_caly_model_devi.RunCalyModelDevi.import_deepmd_package.calc_model_devi")
    # @patch("dpgen2.op.run_caly_model_devi.RunCalyModelDevi.import_deepmd_package.DP")
    @unittest.skipIf(x == 1, "deepmd package not exists.")
    @patch("deepmd.infer.calc_model_devi")
    @patch("deepmd.infer.DeepPot")
    def test_02_success(self, mocked_run_1, mocked_run_2):
        def side_effect_1(*args, **kwargs):
            return "foo"

        mocked_run_1.side_effect = side_effect_1

        def side_effect_2(*args, **kwargs):
            return [[1, 1, 1, 1, 1, 1, 1, 1]]

        mocked_run_2.side_effect = side_effect_2

        op = RunCalyModelDevi()
        out = op.execute(
            OPIO(
                {
                    "type_map": self.type_map,
                    "task_name": str(self.task_name),
                    "traj_dirs": [self.work_dir],
                    "models": self.models,
                }
            )
        )
        # check output
        self.assertEqual(len(out["traj"]), 8)
        self.assertTrue(
            self.task_name / "traj.0.dump" in out["traj"],
        )
        self.assertTrue(
            self.task_name / "traj.1.dump" in out["traj"],
        )
        self.assertTrue(
            self.task_name / "traj.2.dump" in out["traj"],
        )
        (
            self.assertTrue(
                self.task_name / "traj.5.dump" in out["traj"],
            ),
        )
        self.assertTrue(
            self.task_name / "traj.7.dump" in out["traj"],
        )

        self.assertEqual(len(out["model_devi"]), 8)
        self.assertTrue(
            self.task_name / "model_devi.0.out" in out["model_devi"],
        )
        self.assertTrue(
            self.task_name / "model_devi.1.out" in out["model_devi"],
        )
        self.assertTrue(
            self.task_name / "model_devi.2.out" in out["model_devi"],
        )
        self.assertTrue(
            self.task_name / "model_devi.7.out" in out["model_devi"],
        )
