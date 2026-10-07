import json
import os
import unittest
from pathlib import (
    Path,
)

import dpdata
import h5py
import numpy as np
from dflow.python import (
    FatalError,
)
from dflow.python.opio import (
    HDF5Dataset,
)

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.exploration.render import TrajRenderLammps
from dpgen2.exploration.deviation import DeviManager

# isort: on


class TestTrajRenderLammps(unittest.TestCase):
    deviation_names = (
        DeviManager.MAX_DEVI_V,
        DeviManager.MIN_DEVI_V,
        DeviManager.AVG_DEVI_V,
        DeviManager.MAX_DEVI_F,
        DeviManager.MIN_DEVI_F,
        DeviManager.AVG_DEVI_F,
    )

    def test_rejects_non_finite_model_deviation(self):
        """Reject NaN and either infinity in every exported column."""
        model_devi = Path("model_devi.out")
        for column, name in enumerate(self.deviation_names, start=1):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(column=name, value=value):
                    data = np.zeros((2, 7))
                    data[1, column] = value
                    np.savetxt(model_devi, data)
                    with self.assertRaisesRegex(
                        FatalError,
                        rf"Non-finite model-deviation value in model_devi.out: row 2 \({name}\)",
                    ):
                        TrajRenderLammps().get_model_devi([model_devi])

    def test_rejects_non_finite_hdf5_model_deviation(self):
        """An HDF5 error names the dataset, not its Python object address."""
        key = "task.000000/model_devi"
        for column, name in enumerate(self.deviation_names, start=1):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(column=name, value=value):
                    data = np.zeros((2, 7))
                    data[1, column] = value
                    with h5py.File("model_devi.h5", "w") as h5:
                        h5.create_dataset(key, data=data)
                        dataset = HDF5Dataset(h5, key)
                        with self.assertRaises(FatalError) as error:
                            TrajRenderLammps().get_model_devi([dataset])
                    message = str(error.exception)
                    self.assertIn(key, message)
                    self.assertIn(f"row 2 ({name})", message)
                    self.assertNotIn("object at", message)

    def test_limits_non_finite_diagnostics(self):
        """Large failed trajectories give a bounded message and total count."""
        model_devi = Path("model_devi.out")
        data = np.full((20000, 7), np.nan)
        data[:, 0] = np.arange(len(data))
        np.savetxt(model_devi, data)
        with self.assertRaises(FatalError) as error:
            TrajRenderLammps().get_model_devi([model_devi])
        message = str(error.exception)
        self.assertEqual(message.count("row "), 10)
        self.assertIn("120000 non-finite values in total", message)
        self.assertLess(len(message), 1000)

    def test_loads_finite_model_deviation(self):
        """Keep single/multiple rows and large finite sentinel values valid."""
        for nrows in (1, 2):
            for use_hdf5 in (False, True):
                with self.subTest(nrows=nrows, use_hdf5=use_hdf5):
                    data = np.arange(nrows * 7, dtype=float).reshape(nrows, 7)
                    data[0, 4] = np.finfo(float).max
                    if use_hdf5:
                        with h5py.File("model_devi.h5", "w") as h5:
                            h5.create_dataset(
                                "model_devi", data=data[0] if nrows == 1 else data
                            )
                            source = HDF5Dataset(h5, "model_devi")
                            deviations = TrajRenderLammps().get_model_devi([source])
                    else:
                        source = Path("model_devi.out")
                        np.savetxt(source, data)
                        deviations = TrajRenderLammps().get_model_devi([source])
                    for column, name in enumerate(self.deviation_names, start=1):
                        np.testing.assert_array_equal(
                            deviations.get(name)[0], data[:, column]
                        )

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
        for filename in ("job.json", "model_devi.out", "model_devi.h5"):
            if os.path.exists(filename):
                os.remove(filename)
