import glob
import os
import shutil
import unittest

import dpdata
import numpy as np

# isort: off
from dpgen2.utils import (
    setup_ele_temp,
)

# isort: on


class TestSetupEleTemp(unittest.TestCase):
    def test_setup_ele_temp_unlabeled(self):
        system = dpdata.System(
            data={
                "atom_names": ["H"],
                "atom_numbs": [1],
                "atom_types": np.zeros(1, dtype=int),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 1, 3)),
                "orig": np.zeros(3),
                "nopbc": True,
            }
        )
        setup_ele_temp(False)
        system.data["fparam"] = np.array([[1.0]])
        system.to_deepmd_npy("ele_temp_data")
        self.assertEqual(len(glob.glob("ele_temp_data/*/fparam.npy")), 1)
        new_system = dpdata.System("ele_temp_data", fmt="deepmd/npy")
        self.assertTrue("fparam" in new_system.data)

    def test_setup_ele_temp_mixed(self):
        system = dpdata.System(
            data={
                "atom_names": ["H"],
                "atom_numbs": [1],
                "atom_types": np.zeros(1, dtype=int),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 1, 3)),
                "orig": np.zeros(3),
                "nopbc": True,
            }
        )
        setup_ele_temp(True)
        system.data["aparam"] = np.array([[[1.0]]])
        system.to_deepmd_npy_mixed("ele_temp_mixed_data")
        self.assertEqual(len(glob.glob("ele_temp_mixed_data/*/aparam.npy")), 1)
        ms = dpdata.MultiSystems()
        ms.load_systems_from_file(
            "ele_temp_mixed_data", fmt="deepmd/npy/mixed", labeled=False
        )
        self.assertTrue("aparam" in ms[0].data)

    def tearDown(self):
        if os.path.exists("ele_temp_data"):
            shutil.rmtree("ele_temp_data")
        if os.path.exists("ele_temp_mixed_data"):
            shutil.rmtree("ele_temp_mixed_data")
