import json
import os
import unittest

import dpdata
import numpy as np

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.exploration.render import TrajRenderLammps

# isort: on


class TestTrajRenderLammps(unittest.TestCase):
    def test_use_ele_temp_1(self):
        with open("job.json", "w") as f:
            json.dump({"ele_temp": 6.6}, f)
        traj_render = TrajRenderLammps(use_ele_temp=1)
        ele_temp = traj_render.get_ele_temp(["job.json"])
        self.assertEqual(ele_temp, [6.6])

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
        traj_render.set_ele_temp(system, ele_temp[0])
        np.testing.assert_array_almost_equal(system.data["fparam"], np.array([[6.6]]))

    def test_use_ele_temp_2(self):
        with open("job.json", "w") as f:
            json.dump({"ele_temp": 6.6}, f)
        traj_render = TrajRenderLammps(use_ele_temp=2)
        ele_temp = traj_render.get_ele_temp(["job.json"])
        self.assertEqual(ele_temp, [6.6])

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
        traj_render.set_ele_temp(system, ele_temp[0])
        np.testing.assert_array_almost_equal(system.data["aparam"], np.array([[[6.6]]]))

    def tearDown(self):
        if os.path.exists("job.json"):
            os.remove("job.json")
