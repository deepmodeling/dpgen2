import os
import unittest
from pathlib import (
    Path,
)

import numpy as np

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.exploration.deviation import (
    DeviManager,
    DeviManagerSpin,
)

# isort: on


class TestDeviManagerSpin(unittest.TestCase):
    def test_success(self):
        model_devi = DeviManagerSpin()
        model_devi.add(DeviManagerSpin.MAX_DEVI_AF, np.array([1, 2, 3]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_AF, np.array([4, 5, 6]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_MF, np.array([7, 8, 9]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_MF, np.array([10, 11, 12]))

        self.assertEqual(model_devi.ntraj, 2)
        self.assertTrue(
            np.allclose(
                model_devi.get(DeviManagerSpin.MAX_DEVI_AF),
                np.array([[1, 2, 3], [4, 5, 6]]),
            )
        )
        self.assertTrue(
            np.allclose(
                model_devi.get(DeviManagerSpin.MAX_DEVI_MF),
                np.array([[7, 8, 9], [10, 11, 12]]),
            )
        )
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_V), [None, None])

        model_devi.clear()
        self.assertEqual(model_devi.ntraj, 0)
        self.assertEqual(model_devi.get(DeviManagerSpin.MAX_DEVI_AF), [])
        self.assertEqual(model_devi.get(DeviManagerSpin.MAX_DEVI_MF), [])
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_V), [])

    def test_add_invalid_name(self):
        model_devi = DeviManagerSpin()

        self.assertRaisesRegex(
            AssertionError,
            "Error: unknown deviation name foo",
            model_devi.add,
            "foo",
            np.array([1, 2, 3]),
        )

    def test_add_invalid_deviation(self):
        model_devi = DeviManagerSpin()

        self.assertRaisesRegex(
            AssertionError,
            "Error: deviation\(shape: ",
            model_devi.add,
            DeviManagerSpin.MAX_DEVI_AF,
            np.array([[1], [2], [3]]),
        )

        self.assertRaisesRegex(
            AssertionError,
            "Error: deviation\(type: ",
            model_devi.add,
            DeviManagerSpin.MAX_DEVI_MF,
            "foo",
        )

    def test_devi_manager_spin_check_data(self):
        model_devi = DeviManagerSpin()
        model_devi.add(DeviManagerSpin.MAX_DEVI_AF, np.array([1, 2, 3]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_AF, np.array([4, 5, 6]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_MF, np.array([7, 8, 9]))

        self.assertEqual(model_devi.ntraj, 2)

        self.assertRaisesRegex(
            AssertionError,
            "Error: the number of model deviation",
            model_devi.get,
            DeviManagerSpin.MAX_DEVI_MF,
        )

        model_devi = DeviManagerSpin()
        model_devi.add(DeviManagerSpin.MAX_DEVI_MF, np.array([1, 2, 3]))

        self.assertRaisesRegex(
            AssertionError,
            f"Error: cannot find model deviation {DeviManagerSpin.MAX_DEVI_AF}",
            model_devi.get,
            DeviManagerSpin.MAX_DEVI_MF,
        )

        model_devi = DeviManagerSpin()
        model_devi.add(DeviManagerSpin.MAX_DEVI_AF, np.array([1, 2, 3]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_AF, np.array([4, 5, 6]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_MF, np.array([1, 2, 3]))
        model_devi.add(DeviManagerSpin.MAX_DEVI_MF, np.array([4, 5]))
        self.assertRaisesRegex(
            AssertionError,
            f"Error: the number of frames in",
            model_devi.get,
            DeviManagerSpin.MAX_DEVI_AF,
        )
        self.assertRaisesRegex(
            AssertionError,
            f"Error: the number of frames in",
            model_devi.get,
            DeviManagerSpin.MAX_DEVI_MF,
        )
