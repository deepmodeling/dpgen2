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
    DeviManagerStd
)

class TestDeviManager(unittest.TestCase):
    def test_devi_manager_std(self):
        model_devi = DeviManagerStd()
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([1,2,3]))
        model_devi.add(DeviManager.MAX_DEVI_F, np.array([4,5,6]))
        
        self.assertEqual(model_devi.ntraj, 2)
        self.assertTrue(np.allclose(model_devi.get(DeviManager.MAX_DEVI_F), np.array([[1,2,3], [4,5,6]])))
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_V), [None, None])
        
        model_devi.clear()
        self.assertEqual(model_devi.ntraj, 0)
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_F), [])
        self.assertEqual(model_devi.get(DeviManager.MAX_DEVI_V), [])
