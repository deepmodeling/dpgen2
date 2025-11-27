import os
import pickle
import shutil
import sys
import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import numpy as np
from dflow.python import (
    OPIO,
)

from dpgen2.exploration.task import (
    DiffCSPTaskGroup,
)
from dpgen2.op import (
    RunRelax,
)

type_map = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


class DeepPot:
    def __init__(self, *args, **kwargs):
        pass

    def get_type_map(self):
        return type_map

    def eval(*args, **kwargs):
        e = np.array([[-24.51994085]])
        f = np.array(
            [
                [
                    [0.09876919, 0.05018996, -0.38124716],
                    [-0.08633465, -0.05225886, 0.32088113],
                    [-0.00362617, -0.00345137, 0.011181],
                    [-0.00880837, 0.00552027, 0.04918503],
                ]
            ]
        )
        v = np.array(
            [
                [
                    -2.35308634,
                    0.02806883,
                    -0.31607685,
                    0.02806883,
                    -2.29484913,
                    -0.15927731,
                    -0.31607685,
                    -0.15927731,
                    -1.31155533,
                ]
            ]
        )
        return e, f, v


def calc_model_devi_f(fs):
    fs_devi = np.linalg.norm(np.std(fs, axis=0), axis=-1)
    max_devi_f = np.max(fs_devi, axis=-1)
    min_devi_f = np.min(fs_devi, axis=-1)
    avg_devi_f = np.mean(fs_devi, axis=-1)
    return max_devi_f, min_devi_f, avg_devi_f


def calc_model_devi_v(vs):
    vs_devi = np.std(vs, axis=0)
    max_devi_v = np.max(vs_devi, axis=-1)
    min_devi_v = np.min(vs_devi, axis=-1)
    avg_devi_v = np.linalg.norm(vs_devi, axis=-1) / 3
    return max_devi_v, min_devi_v, avg_devi_v


def relax_run(*args, **kwargs):
    os.makedirs("relax_trajs", exist_ok=True)
    with open("relax_trajs/0", "wb") as f:
        pickle.dump(
            {
                "energy": [-24.48511280170925],
                "forces": [
                    np.array(
                        [
                            [0.07237937, 0.0548471, -0.25049795],
                            [-0.05103608, -0.04855114, 0.14486373],
                            [-0.00699195, -0.00202532, 0.05028293],
                            [-0.01435133, -0.00427063, 0.05535129],
                        ]
                    )
                ],
                "stresses": [
                    np.array(
                        [
                            0.02573293,
                            0.02483516,
                            0.02152565,
                            -0.00011215,
                            0.0011876,
                            0.00021871,
                        ]
                    )
                ],
                "atom_positions": [
                    np.array(
                        [
                            [5.16042463, 2.95461897, 4.22479426],
                            [2.10195627, 1.11158311, 8.10963874],
                            [3.62923674, 2.02945063, 2.22314751],
                            [0.81206749, 0.32068803, 5.12969607],
                        ]
                    )
                ],
                "cell": [
                    np.array(
                        [
                            [3.84691675, 0.0, 1.01375962],
                            [1.79422664, 3.40209201, 0.99903783],
                            [0.0, 0.0, 7.87117147],
                        ]
                    )
                ],
                "atomic_number": np.array([57, 57, 6, 35]),
            },
            f,
        )


class DPCalculator:
    def __init__(self, dp):
        self.dp = dp


class Relaxer:
    def __init__(self, *args, **kwargs):
        dp = DeepPot()
        self.calculator = DPCalculator(dp)


class TestRunRelax(unittest.TestCase):
    @patch("dpgen2.op.run_relax.atoms2lmpdump")
    def testRunRelax(self, mocked_run):
        mocked_run.side_effect = ["ITEM: TIMESTEP"]
        sys.modules["deepmd.infer"] = sys.modules[__name__]
        sys.modules["deepmd.infer.model_devi"] = sys.modules[__name__]
        sys.modules["lam_optimize.main"] = sys.modules[__name__]
        sys.modules["lam_optimize.relaxer"] = sys.modules[__name__]

        task_group = DiffCSPTaskGroup()
        task_group.make_task()
        os.makedirs("task.000000", exist_ok=True)
        op_in = OPIO(
            {
                "diffcsp_task_grp": task_group,
                "expl_config": {},
                "task_path": Path("task.000000"),
                "models": [Path("model_0.pt"), Path("model_1.pt")],
            }
        )
        op = RunRelax()
        op_out = op.execute(op_in)
        self.assertEqual(len(op_out["trajs"]), 1)
        self.assertTrue(op_out["trajs"][0].read_text().startswith("ITEM: TIMESTEP"))
        self.assertEqual(len(op_out["model_devis"]), 1)
        model_devi = np.array(
            [0.0, 0.1132373, 0.00632493, 0.04404319, 0.0897801, 0.006415, 0.04564122]
        )
        np.testing.assert_array_almost_equal(
            np.loadtxt(op_out["model_devis"][0]), model_devi
        )

    def tearDown(self):
        if os.path.isdir("task.000000"):
            shutil.rmtree("task.000000")
        if os.path.isdir("relax_trajs"):
            shutil.rmtree("relax_trajs")
