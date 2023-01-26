from context import dpgen2
import os, textwrap
import numpy as np
import unittest
from collections import Counter
from dpgen2.exploration.report import ExplorationReportTrustLevels
from dargs import Argument

class TestTrajsExplorationResport(unittest.TestCase):
    def test_fv(self):
        md_f = [ np.array([ 0.90, 0.10, 0.91, 0.11, 0.50, 0.12, 0.51, 0.52, 0.92 ]),
                 np.array([ 0.40, 0.20, 0.80, 0.81, 0.82, 0.21, 0.41, 0.22, 0.42 ]) ]
        md_v = [ np.array([ 0.40, 0.20, 0.21, 0.80, 0.81, 0.41, 0.22, 0.82, 0.42 ]),
                 np.array([ 0.50, 0.90, 0.91, 0.92, 0.51, 0.52, 0.10, 0.11, 0.12 ]) ]
        # id_f_accu = [ [3, 5, 1], [1, 7, 5] ]
        # id_f_cand = [ [4, 7, 6], [8, 6, 0] ]
        # id_f_fail = [ [2, 0, 8], [4, 2, 3] ]
        # id_v_accu = [ [1, 2, 6], [7, 8, 6] ]
        # id_v_cand = [ [0, 5, 8], [0, 5, 4] ]
        # id_v_fail = [ [4, 3, 7], [2, 3, 1] ]
        expected_accu = [ [1], [7] ]
        expected_cand = [ [6, 5], [8, 6, 0, 5] ]
        expected_fail = [ [0, 2, 3, 4, 7, 8], [1, 2, 3, 4] ]
        expected_accu = [set(ii) for ii in expected_accu]
        expected_cand = [set(ii) for ii in expected_cand]
        expected_fail = [set(ii) for ii in expected_fail]
        all_cand_sel = [ (0,6), (0,5), (1,8), (1,6), (1,0), (1,5) ]

        ter = ExplorationReportTrustLevels(0.3, 0.6, 0.3, 0.6, conv_accuracy=0.9)
        ter.record(md_f, md_v)
        self.assertFalse(ter.converged())
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
        self.assertEqual(ter.candidate_ratio(), 6./18.)
        self.assertEqual(ter.accurate_ratio(), 2./18.)
        self.assertEqual(ter.failed_ratio(), 10./18.)


    def test_f(self):
        md_f = [ np.array([ 0.90, 0.10, 0.91, 0.11, 0.50, 0.12, 0.51, 0.52, 0.92 ]),
                 np.array([ 0.40, 0.20, 0.80, 0.81, 0.82, 0.21, 0.41, 0.22, 0.42 ]) ]
        md_v = None
        id_f_accu = [ [3, 5, 1], [1, 7, 5] ]
        id_f_cand = [ [4, 7, 6], [8, 6, 0] ]
        id_f_fail = [ [2, 0, 8], [4, 2, 3] ]
        # id_v_accu = [None, None]
        # id_v_cand = [None, None]
        # id_v_fail = [None, None]
        expected_accu = id_f_accu
        expected_cand = id_f_cand
        expected_fail = id_f_fail
        expected_accu = [set(ii) for ii in expected_accu]
        expected_cand = [set(ii) for ii in expected_cand]
        expected_fail = [set(ii) for ii in expected_fail]
        all_cand_sel = [ (0,4), (0,7), (0,6), (1,6), (1,0), (1,8) ]
        
        ter = ExplorationReportTrustLevels(0.3, 0.6, 0.3, 0.6, conv_accuracy=0.2)
        ter.record(md_f, md_v)
        self.assertTrue(ter.converged())
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


    def test_f_max(self):
        md_f = [ np.array([ 0.90, 0.10, 0.91, 0.11, 0.50, 0.12, 0.51, 0.52, 0.92 ]),
                 np.array([ 0.40, 0.20, 0.80, 0.81, 0.82, 0.21, 0.41, 0.22, 0.42 ]) ]
        md_v = None
        id_f_accu = [ [3, 5, 1], [1, 7, 5] ]
        id_f_cand = [ [4, 7, 6], [8, 6, 0] ]
        id_f_fail = [ [2, 0, 8], [4, 2, 3] ]
        # id_v_accu = [None, None]
        # id_v_cand = [None, None]
        # id_v_fail = [None, None]
        expected_accu = id_f_accu
        expected_cand = id_f_cand
        expected_fail = id_f_fail
        expected_accu = [set(ii) for ii in expected_accu]
        expected_cand = [set(ii) for ii in expected_cand]
        expected_fail = [set(ii) for ii in expected_fail]
        all_cand_sel = [ (0,4), (0,7), (0,6), (1,6), (1,0), (1,8) ]
        
        ter = ExplorationReportTrustLevels(0.3, 0.6, 0.3, 0.6, conv_accuracy=0.2)
        ter.record(md_f, md_v)
        self.assertTrue(ter.converged())
        self.assertEqual(ter.traj_cand, expected_cand)
        self.assertEqual(ter.traj_accu, expected_accu)
        self.assertEqual(ter.traj_fail, expected_fail)
        
        picked = ter.get_candidate_ids(10)
        npicked = 0
        self.assertEqual(len(picked), 2)
        for ii in range(2):
            for jj in picked[ii]:
                self.assertTrue(jj in expected_cand[ii])
                npicked += 1
        self.assertEqual(npicked, 6)


    def test_args(self):
        input_dict = {
            "level_f_lo" : 0.5,
            "level_f_hi" : 1.0,
            "conv_accuracy" : 0.9,
        }

        base = Argument("base", dict, ExplorationReportTrustLevels.args())
        data = base.normalize_value(input_dict)
        self.assertAlmostEqual(data['level_f_lo'], .5)
        self.assertAlmostEqual(data['level_f_hi'], 1.)
        self.assertTrue(data['level_v_lo'] is None)
        self.assertTrue(data['level_v_hi'] is None)
        self.assertAlmostEqual(data['conv_accuracy'], 0.9)
        ExplorationReportTrustLevels(*data)
