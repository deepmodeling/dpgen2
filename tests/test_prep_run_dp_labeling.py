import numpy as np
import unittest, os, shutil
from pathlib import Path

from dpgen2.fp.deepmd import (
    deepmd_input_path,
    deepmd_temp_path,
    deepmd_teacher_model,
    PrepDeepmd,
    RunDeepmd,
)
from dflow.python import FatalError
from dpgen2.utils import BinaryFileInput
from dargs import Argument

import dpdata
from mock import patch, Mock
import sys


class TestPrepDeepmd(unittest.TestCase):
    def setUp(self):
        self.system = dpdata.System(
            data={
                "atom_names": ["H"],
                "atom_numbs": [1],
                "atom_types": np.zeros(1, dtype=int),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 1, 3)),
                "orig": np.zeros(3),
            }
        )

    def tearDown(self):
        if Path(deepmd_input_path).is_dir():
            shutil.rmtree(deepmd_input_path)

    def test_prep_task(self):
        prep_deepmd = PrepDeepmd()
        prep_deepmd.prep_task(
            self.system,
            {},
        )
        ss = dpdata.System(deepmd_input_path, fmt="deepmd/npy")
        self.assertEqual(self.system["atom_names"], ss["atom_names"])
        self.assertEqual(self.system["atom_numbs"], ss["atom_numbs"])
        self.assertTrue(np.allclose(self.system["atom_types"], ss["atom_types"]))
        self.assertTrue(np.allclose(self.system["cells"], ss["cells"]))
        self.assertTrue(np.allclose(self.system["coords"], ss["coords"]))


class TestRunDeepmd(unittest.TestCase):
    def setUp(self):
        self.system = dpdata.System(
            data={
                "atom_names": ["H", "O"],
                "atom_numbs": [1, 1],
                "atom_types": np.array([0, 1]),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 2, 3)),
                "orig": np.zeros(3),
            }
        )
        self.task_path = Path("task")
        self.task_path.mkdir(parents=True, exist_ok=True)
        Path(self.task_path / "teacher-model.pb").write_bytes(b"0123456789")
        self.teacher_model = BinaryFileInput(self.task_path / "teacher-model.pb")

    def tearDown(self):
        shutil.rmtree(self.task_path, ignore_errors=True)

        shutil.rmtree(deepmd_input_path, ignore_errors=True)

        shutil.rmtree(deepmd_teacher_model, ignore_errors=True)

        shutil.rmtree(deepmd_teacher_model, ignore_errors=True)

    def test_prep_input(self):
        run_deepmd = RunDeepmd()
        out_name = self.task_path / "test_out"
        self.system.to("deepmd/npy", deepmd_input_path)

        # test1
        self.assertRaisesRegex(
            FatalError, "is not subset", run_deepmd._prep_input, ["O"]
        )

        # test2
        self.assertRaisesRegex(
            FatalError, "is not subset", run_deepmd._prep_input, ["H"]
        )

        # test3
        shutil.rmtree(out_name, ignore_errors=True)
        shutil.rmtree(deepmd_temp_path, ignore_errors=True)
        run_deepmd._prep_input(["H", "O"])

        ss = dpdata.System(deepmd_temp_path, fmt="deepmd/npy")
        self.assertTrue(ss["atom_names"] == ["H", "O"])

        # test4
        shutil.rmtree(out_name, ignore_errors=True)
        shutil.rmtree(deepmd_temp_path, ignore_errors=True)
        run_deepmd._prep_input(["O", "H"])

        ss = dpdata.System(deepmd_temp_path, fmt="deepmd/npy")
        self.assertTrue(ss["atom_names"] == ["O", "H"])

        # test5
        shutil.rmtree(out_name, ignore_errors=True)
        shutil.rmtree(deepmd_temp_path, ignore_errors=True)
        run_deepmd._prep_input(["O", "C", "H", "N"])

        ss = dpdata.System(deepmd_temp_path, fmt="deepmd/npy")
        self.assertTrue(ss["atom_names"] == ["O", "H"])

    def test_get_dp_model(self):
        # from deepmd.infer import DeepPot
        deepmd = Mock()
        modules = {"deepmd": deepmd, "deepmd.infer": deepmd.infer}
        deepmd.infer.DeepPot = Mock()
        dp = deepmd.infer.DeepPot.return_value
        with patch.dict("sys.modules", modules):
            run_deepmd = RunDeepmd()
            # test1
            dp.get_type_map.return_value = ["H", "C"]
            _dp, _type_map = run_deepmd._get_dp_model(self.teacher_model)
            self.assertTrue(_dp is dp)
            self.assertEqual(_type_map, ["H", "C"])
            deepmd.infer.DeepPot.assert_called_once_with(deepmd_teacher_model)
            self.assertFalse(Path(deepmd_teacher_model).is_file())

    def test_dp_infer(self):
        self.system.to("deepmd/npy", deepmd_input_path)

        out_name = self.task_path / "test_out"

        dp = Mock()
        self._set_mock_dp_eval(dp)

        run_deepmd = RunDeepmd()
        run_deepmd._dp_infer(dp, ["H", "O"], str(out_name))

    def test_run_task(self):
        # run prep_task
        prep_deepmd = PrepDeepmd()
        prep_deepmd.prep_task(
            self.system,
            {},
        )

        out_name = self.task_path / "test_out"
        log_name = self.task_path / "test.log"

        prep_deepmd = RunDeepmd()

        # run run_task
        deepmd = Mock()
        modules = {"deepmd": deepmd, "deepmd.infer": deepmd.infer}
        deepmd.infer.DeepPot = Mock()
        dp = deepmd.infer.DeepPot.return_value
        self._set_mock_dp_eval(dp)
        with patch.dict("sys.modules", modules):
            prep_deepmd.run_task(
                self.teacher_model,
                str(out_name),
                str(log_name),
            )
            self.assertTrue(log_name.is_file())
            self.assertTrue(out_name.is_dir())
            self._check_output_system(out_name)

    def _set_mock_dp_eval(self, dp):
        energy, force, virial_foce = self._get_labels()
        dp.eval.return_value = (energy, force, virial_foce)
        dp.get_type_map.return_value = ["O", "H"]

    def _check_output_system(self, path):
        ss = dpdata.LabeledSystem(path, fmt="deepmd/npy")
        energy, force, virial_foce = self._get_labels()

        self.assertTrue(ss["atom_numbs"] == self.system["atom_numbs"])
        self.assertTrue(np.allclose(ss["atom_types"], self.system["atom_types"]))
        self.assertTrue(np.allclose(ss["energies"], energy))
        self.assertTrue(np.allclose(ss["forces"], force))
        self.assertTrue(np.allclose(ss["virials"], virial_foce))

    def _get_labels(self):
        energy = np.zeros(self.system["coords"].shape[0])
        force = np.zeros_like(self.system["coords"])
        virial_foce = np.zeros((self.system["coords"].shape[0], 3, 3))
        return energy, force, virial_foce
