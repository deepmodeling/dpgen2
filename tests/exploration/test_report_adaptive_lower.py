from context import dpgen2
import os, textwrap
import numpy as np
import unittest
from collections import Counter
from dpgen2.exploration.report import ExplorationReportAdaptiveLower
from dargs import Argument


class TestTrajsExplorationResport(unittest.TestCase):
    def test_fv(self):
        md_f = [
            np.array([0.90, 0.10, 0.91, 0.11, 0.50, 0.53, 0.51, 0.52, 0.92]),
            np.array([0.40, 0.20, 0.80, 0.81, 0.82, 0.21, 0.41, 0.22, 0.42]),
        ]
        md_v = [
            np.array([0.40, 0.20, 0.21, 0.80, 0.81, 0.53, 0.22, 0.82, 0.42]),
            np.array([0.50, 0.90, 0.91, 0.92, 0.51, 0.52, 0.10, 0.11, 0.12]),
        ]
        expected_fail_ = [[0, 2, 3, 4, 7, 8], [1, 2, 3, 4]]
        expected_fail = set()
        for idx, ii in enumerate(expected_fail_):
            for jj in ii:
                expected_fail.add((idx, jj))
        expected_cand = set([(0, 5), (0, 6), (1, 8), (1, 0), (1, 5)])
        expected_accu = set([(0, 1), (1, 6), (1, 7)])

        ter = ExplorationReportAdaptiveLower(
            level_f_hi=0.7,
            numb_candi_f=3,
            rate_candi_f=0.001,
            level_v_hi=0.76,
            numb_candi_v=3,
            rate_candi_v=0.001,
            n_checked_steps=3,
            conv_tolerance=0.001,
        )
        ter.record(md_f, md_v)
        self.assertEqual(ter.candi, expected_cand)
        self.assertEqual(ter.accur, expected_accu)
        self.assertEqual(set(ter.failed), expected_fail)

        class MockedReport:
            level_f_lo = 0
            level_v_lo = 0

        mr = MockedReport()
        mr.level_f_lo = 0.42
        mr.level_v_lo = 0.50
        self.assertFalse(ter.converged([mr]))
        self.assertTrue(ter.converged([mr, mr]))
        self.assertTrue(ter.converged([mr, mr, mr]))
        mr1 = MockedReport()
        mr1.level_f_lo = 0.42
        mr1.level_v_lo = 0.60
        self.assertFalse(ter.converged([mr]))
        self.assertFalse(ter.converged([mr, mr1]))
        self.assertFalse(ter.converged([mr, mr1, mr]))
        self.assertTrue(ter.converged([mr1, mr, mr]))

        picked = ter.get_candidate_ids(2)
        npicked = 0
        self.assertEqual(len(picked), 2)
        for ii in range(2):
            for jj in picked[ii]:
                self.assertTrue((ii, jj) in expected_cand)
                npicked += 1
        self.assertEqual(npicked, 2)

        picked = ter.get_candidate_ids(10)
        npicked = 0
        self.assertEqual(len(picked), 2)
        for ii in range(2):
            for jj in picked[ii]:
                self.assertTrue((ii, jj) in expected_cand)
                npicked += 1
        self.assertEqual(npicked, 5)

        self.assertEqual(ter.candidate_ratio(), 5.0 / 18.0)
        self.assertEqual(ter.accurate_ratio(), 3.0 / 18.0)
        self.assertEqual(ter.failed_ratio(), 10.0 / 18.0)

        self.assertEqual(
            ter.print_header(),
            "#   stage  id_stg.    iter.      accu.      cand.      fail.   lvl_f_lo   lvl_f_hi       v_lo     v_hi",
        )
        self.assertEqual(
            ter.print(0, 1, 1),
            "        0        1        1     0.1667     0.2778     0.5556     0.4200     0.7000     0.5000   0.7600",
        )

    def test_f(self):
        md_f = [
            np.array([0.90, 0.10, 0.91, 0.11, 0.50, 0.53, 0.51, 0.52, 0.92]),
            np.array([0.40, 0.20, 0.80, 0.81, 0.82, 0.21, 0.41, 0.22, 0.42]),
        ]
        expected_fail_ = [[0, 2, 8], [2, 3, 4]]
        expected_fail = set()
        for idx, ii in enumerate(expected_fail_):
            for jj in ii:
                expected_fail.add((idx, jj))
        expected_cand = set([(0, 6), (0, 7), (0, 5)])
        expected_accu = set(
            [(0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (1, 5), (1, 6), (1, 7), (1, 8)]
        )

        ter = ExplorationReportAdaptiveLower(
            level_f_hi=0.7,
            numb_candi_f=3,
            rate_candi_f=0.001,
            n_checked_steps=2,
            conv_tolerance=0.1,
        )
        ter.record(md_f)
        self.assertFalse(ter.converged([]))
        self.assertEqual(ter.candi, expected_cand)
        self.assertEqual(ter.accur, expected_accu)
        self.assertEqual(set(ter.failed), expected_fail)

        picked = ter.get_candidate_ids(2)
        npicked = 0
        self.assertEqual(len(picked), 2)
        for ii in range(2):
            for jj in picked[ii]:
                self.assertTrue((ii, jj) in expected_cand)
                npicked += 1
        self.assertEqual(npicked, 2)
        self.assertEqual(ter.candidate_ratio(), 3.0 / 18.0)
        self.assertEqual(ter.accurate_ratio(), 9.0 / 18.0)
        self.assertEqual(ter.failed_ratio(), 6.0 / 18.0)

    def test_v(self):
        md_v = [
            np.array([0.90, 0.10, 0.91, 0.11, 0.50, 0.53, 0.51, 0.52, 0.92]),
            np.array([0.40, 0.20, 0.80, 0.81, 0.82, 0.21, 0.41, 0.22, 0.42]),
        ]
        md_f = [
            np.array([0.40, 0.20, 0.21, 0.80, 0.81, 0.53, 0.22, 0.82, 0.42]),
            np.array([0.50, 0.90, 0.91, 0.92, 0.51, 0.52, 0.10, 0.11, 0.12]),
        ]
        expected_fail_ = [[0, 2, 8], [2, 3, 4]]
        expected_fail = set()
        for idx, ii in enumerate(expected_fail_):
            for jj in ii:
                expected_fail.add((idx, jj))
        expected_cand = set([(0, 6), (0, 7), (0, 5)])
        expected_accu = set(
            [(0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (1, 5), (1, 6), (1, 7), (1, 8)]
        )

        ter = ExplorationReportAdaptiveLower(
            level_f_hi=1.0,
            numb_candi_f=0,
            rate_candi_f=0.000,
            level_v_hi=0.76,
            numb_candi_v=3,
            rate_candi_v=0.001,
            n_checked_steps=3,
            conv_tolerance=0.001,
        )
        ter.record(md_f, md_v)
        self.assertEqual(ter.candi, expected_cand)
        self.assertEqual(ter.accur, expected_accu)
        self.assertEqual(set(ter.failed), expected_fail)

        class MockedReport:
            level_f_lo = 0
            level_v_lo = 0

        mr = MockedReport()
        mr.level_f_lo = 1.0
        mr.level_v_lo = 0.51
        self.assertFalse(ter.converged([mr]))
        self.assertTrue(ter.converged([mr, mr]))
        self.assertTrue(ter.converged([mr, mr, mr]))

        picked = ter.get_candidate_ids(2)
        npicked = 0
        self.assertEqual(len(picked), 2)
        for ii in range(2):
            for jj in picked[ii]:
                self.assertTrue((ii, jj) in expected_cand)
                npicked += 1
        self.assertEqual(npicked, 2)
        self.assertEqual(ter.candidate_ratio(), 3.0 / 18.0)
        self.assertEqual(ter.accurate_ratio(), 9.0 / 18.0)
        self.assertEqual(ter.failed_ratio(), 6.0 / 18.0)

    def test_args(self):
        input_dict = {
            "level_f_hi": 1.0,
            "numb_candi_f": 100,
            "rate_candi_f": 0.1,
            "conv_tolerance": 0.01,
        }

        base = Argument("base", dict, ExplorationReportAdaptiveLower.args())
        data = base.normalize_value(input_dict)
        self.assertAlmostEqual(data["level_f_hi"], 1.0)
        self.assertEqual(data["numb_candi_f"], 100)
        self.assertAlmostEqual(data["rate_candi_f"], 0.1)
        self.assertTrue(data["level_v_hi"] is None)
        self.assertEqual(data["numb_candi_v"], 0)
        self.assertAlmostEqual(data["rate_candi_v"], 0.0)
        self.assertEqual(data["n_checked_steps"], 2)
        self.assertAlmostEqual(data["conv_tolerance"], 0.01)
        ExplorationReportAdaptiveLower(*data)
