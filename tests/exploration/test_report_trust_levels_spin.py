import os
import textwrap
import unittest
from collections import (
    Counter,
)

import numpy as np
from dargs import (
    Argument,
)

# isort: off
from context import (
    dpgen2,
)
from dpgen2.exploration.deviation import (
    DeviManager,
    DeviManagerSpin,
)
from dpgen2.exploration.report import (
    ExplorationReportTrustLevelsSpin,
)
from dpgen2.exploration.report.report_trust_levels_base import (
    ExplorationReportTrustLevels,
)

# isort: on


class TestTrajsExplorationReportSpin(unittest.TestCase):
    def test_exploration_report_trust_levels_spin(self):
        self.selection_test(ExplorationReportTrustLevelsSpin)
        self.args_test(ExplorationReportTrustLevelsSpin)

    def selection_test(self, exploration_report: ExplorationReportTrustLevelsSpin):
        model_devi = DeviManagerSpin()
        model_devi.add(
            DeviManagerSpin.MAX_DEVI_AF,
            np.array([0.90, 0.10, 0.91, 0.11, 0.50, 0.12, 0.51, 0.52, 0.92]),
        )
        model_devi.add(
            DeviManagerSpin.MAX_DEVI_AF,
            np.array([0.40, 0.20, 0.80, 0.81, 0.82, 0.21, 0.41, 0.22, 0.42]),
        )
        model_devi.add(
            DeviManagerSpin.MAX_DEVI_MF,
            np.array([0.40, 0.20, 0.21, 0.80, 0.81, 0.41, 0.22, 0.82, 0.42]),
        )
        model_devi.add(
            DeviManagerSpin.MAX_DEVI_MF,
            np.array([0.50, 0.90, 0.91, 0.92, 0.51, 0.52, 0.10, 0.11, 0.12]),
        )

        # id_f_accu = [ [3, 5, 1], [1, 7, 5] ]
        # id_f_cand = [ [4, 7, 6], [8, 6, 0] ]
        # id_f_fail = [ [2, 0, 8], [4, 2, 3] ]
        # id_v_accu = [ [1, 2, 6], [7, 8, 6] ]
        # id_v_cand = [ [0, 5, 8], [0, 5, 4] ]
        # id_v_fail = [ [4, 3, 7], [2, 3, 1] ]
        expected_accu = [[1], [7]]
        expected_cand = [[6, 5], [8, 6, 0, 5]]
        expected_fail = [[0, 2, 3, 4, 7, 8], [1, 2, 3, 4]]
        expected_accu = [set(ii) for ii in expected_accu]
        expected_cand = [set(ii) for ii in expected_cand]
        expected_fail = [set(ii) for ii in expected_fail]
        all_cand_sel = [(0, 6), (0, 5), (1, 8), (1, 6), (1, 0), (1, 5)]

        ter = exploration_report(0.3, 0.6, 0.3, 0.6, conv_accuracy=0.9)
        ter.record(model_devi)
        self.assertEqual(ter.traj_cand, expected_cand)
        self.assertEqual(ter.traj_accu, expected_accu)
        self.assertEqual(ter.traj_fail, expected_fail)

        picked = ter.get_candidate_ids(2)
        npicked = 0
        self.assertEqual(len(picked), 2)
        for ii in range(2):
            for jj in picked[ii]:
                self.assertTrue(jj in expected_cand[ii])
                npicked += 1
        self.assertEqual(npicked, 2)
        self.assertEqual(ter.candidate_ratio(), 6.0 / 18.0)
        self.assertEqual(ter.accurate_ratio(), 2.0 / 18.0)
        self.assertEqual(ter.failed_ratio(), 10.0 / 18.0)

    def args_test(self, exploration_report: ExplorationReportTrustLevelsSpin):
        input_dict = {
            "level_af_lo": 0.5,
            "level_af_hi": 1.0,
            "level_mf_lo": 0.05,
            "level_mf_hi": 0.1,
            "conv_accuracy": 0.9,
        }

        base = Argument("base", dict, exploration_report.args())
        data = base.normalize_value(input_dict)
        self.assertAlmostEqual(data["level_af_lo"], 0.5)
        self.assertAlmostEqual(data["level_af_hi"], 1.0)
        self.assertAlmostEqual(data["level_mf_lo"], 0.05)
        self.assertAlmostEqual(data["level_mf_hi"], 0.1)
        self.assertAlmostEqual(data["conv_accuracy"], 0.9)
        exploration_report(*data)