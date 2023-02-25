import os
import unittest
from pathlib import (
    Path,
)

import numpy as np
from context import (
    dpgen2,
)

from dpgen2.exploration.deviation import (
    DeviManager,
    DeviManagerStd,
)


class TestDeviManagerStd(unittest.TestCase):
    def test_success(self):
        model_devi = DeviManagerStd()
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([1, 2, 3]))
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([4, 5, 6]))

        self.assertEqual(model_devi.ntraj, 2)
        self.assertTrue(
            np.allclose(
                model_devi.get(DeviManager.MAX_DEVI_F), np.array([[1, 2, 3], [4, 5, 6]])
            )
        )
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_V), [None, None])

        model_devi.clear()
        self.assertEqual(model_devi.ntraj, 0)
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_F), [])
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_V), [])

    def test_add_invalid_name(self):
        model_devi = DeviManagerStd()

        self.assertRaisesRegex(
            AssertionError,
            "Error: unknown deviation name foo",
            model_devi.add,
            "foo",
            np.array([1, 2, 3]),
        )

    def test_add_invalid_deviation(self):
        model_devi = DeviManagerStd()

        self.assertRaisesRegex(
            AssertionError,
            "Error: deviation\(shape: ",
            model_devi.add,
            DeviManager.MAX_DEVI_F,
            np.array([[1], [2], [3]]),
        )

        self.assertRaisesRegex(
            AssertionError,
            "Error: deviation\(type: ",
            model_devi.add,
            DeviManager.MAX_DEVI_F,
            "foo",
        )

    def test_devi_manager_std_check_data(self):
        model_devi = DeviManagerStd()
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([1, 2, 3]))
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([4, 5, 6]))
        model_devi.add(DeviManager.MAX_DEVI_V, np.array([4, 5, 6]))

        self.assertEqual(model_devi.ntraj, 2)

        self.assertRaisesRegex(
            AssertionError,
            "Error: the number of model deviation",
            model_devi.get,
            DeviManager.MAX_DEVI_V,
        )

        model_devi = DeviManagerStd()
        model_devi.add(DeviManager.MAX_DEVI_V, np.array([1, 2, 3]))

        self.assertRaisesRegex(
            AssertionError,
            f"Error: cannot find model deviation {DeviManager.MAX_DEVI_F}",
            model_devi.get,
            DeviManager.MAX_DEVI_V,
        )

        model_devi = DeviManagerStd()
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([1, 2, 3]))
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([4, 5, 6]))
        model_devi.add(DeviManager.MAX_DEVI_V, np.array([1, 2, 3]))
        model_devi.add(DeviManager.MAX_DEVI_V, np.array([4, 5]))
        self.assertRaisesRegex(
            AssertionError,
            f"Error: the number of frames in",
            model_devi.get,
            DeviManager.MAX_DEVI_F,
        )
        self.assertRaisesRegex(
            AssertionError,
            f"Error: the number of frames in",
            model_devi.get,
            DeviManager.MAX_DEVI_V,
        )
