from context import dpgen2
import os, textwrap
import numpy as np
import unittest
from collections import Counter
from dpgen2.exploration.report import NaiveExplorationReport, TrajsExplorationReport

class TestNaiveExplorationReport(unittest.TestCase):
    def test_naive_fv(self):
        counter_f = Counter()
        counter_v = Counter()
        counter_f['candidate'] = 5
        counter_f['accurate'] = 4
        counter_f['failed'] = 1
        counter_v['candidate'] = 7
        counter_v['accurate'] = 2
        counter_v['failed'] = 1
        report = NaiveExplorationReport(counter_f, counter_v)
        self.assertAlmostEqual(report.ratio('force', 'candidate'), 0.5)
        self.assertAlmostEqual(report.ratio('force', 'accurate'), 0.4)
        self.assertAlmostEqual(report.ratio('force', 'failed'), 0.1)
        self.assertAlmostEqual(report.ratio('virial', 'candidate'), 0.7)
        self.assertAlmostEqual(report.ratio('virial', 'accurate'), 0.2)
        self.assertAlmostEqual(report.ratio('virial', 'failed'), 0.1)
        self.assertAlmostEqual(report.candidate_ratio(), 0.5)
        self.assertAlmostEqual(report.accurate_ratio(), 0.4)
        self.assertAlmostEqual(report.failed_ratio(), 0.1)
        

    def test_naive_f(self):
        counter_f = Counter()
        counter_v = Counter()
        counter_f['candidate'] = 5
        counter_f['accurate'] = 4
        counter_f['failed'] = 1
        counter_v['candidate'] = 0
        counter_v['accurate'] = 0
        counter_v['failed'] = 0
        report = NaiveExplorationReport(counter_f, counter_v)
        self.assertAlmostEqual(report.ratio('force', 'candidate'), 0.5)
        self.assertAlmostEqual(report.ratio('force', 'accurate'), 0.4)
        self.assertAlmostEqual(report.ratio('force', 'failed'), 0.1)
        self.assertAlmostEqual(report.ratio('virial', 'candidate'), None)
        self.assertAlmostEqual(report.ratio('virial', 'accurate'), None)
        self.assertAlmostEqual(report.ratio('virial', 'failed'), None)
        self.assertAlmostEqual(report.candidate_ratio(), 0.5)
        self.assertAlmostEqual(report.accurate_ratio(), 0.4)
        self.assertAlmostEqual(report.failed_ratio(), 0.1)


    def test_naive_failed(self):
        counter_f = Counter()
        counter_v = Counter()
        counter_f['candidate'] = 5
        counter_f['accurate'] = 4
        counter_f['failed'] = 1
        counter_v['candidate'] = 7
        counter_v['accurate'] = 2
        counter_v['failed'] = 1
        report = NaiveExplorationReport(counter_f, counter_v)
        with self.assertRaises(RuntimeError) as context:
            report.ratio('foo', 'candidate')
        self.assertTrue('invalid quantity foo' in str(context.exception))
        with self.assertRaises(RuntimeError) as context:
            report.ratio('force', 'bar')
        self.assertTrue('invalid item bar' in str(context.exception))


class TestTrajsExplorationResport(unittest.TestCase):
    def test_fv(self):
        id_f_accu = [ [3, 5, 1], [1, 7, 5] ]
        id_f_cand = [ [4, 7, 6], [8, 6, 0] ]
        id_f_fail = [ [2, 0, 8], [4, 2, 3] ]
        id_v_accu = [ [1, 2, 6], [7, 8, 6] ]
        id_v_cand = [ [0, 5, 8], [0, 5, 4] ]
        id_v_fail = [ [4, 3, 7], [2, 3, 1] ]
        expected_accu = [ [1], [7] ]
        expected_cand = [ [6, 5], [8, 6, 0, 5] ]
        expected_fail = [ [0, 2, 3, 4, 7, 8], [1, 2, 3, 4] ]
        expected_accu = [set(ii) for ii in expected_accu]
        expected_cand = [set(ii) for ii in expected_cand]
        expected_fail = [set(ii) for ii in expected_fail]
        all_cand_sel = [ (0,6), (0,5), (1,8), (1,6), (1,0), (1,5) ]
        
        ter = TrajsExplorationReport()
        for ii in range(2):
            ter.record_traj(
                id_f_accu[ii], id_f_cand[ii], id_f_fail[ii],
                id_v_accu[ii], id_v_cand[ii], id_v_fail[ii],
            )
        self.assertEqual(ter.traj_cand, expected_cand)
        self.assertEqual(ter.traj_accu, expected_accu)
        self.assertEqual(ter.traj_fail, expected_fail)
        
        picked = ter.get_candidates(2)
        self.assertEqual(len(picked), 2)
        self.assertTrue(picked[0] in all_cand_sel)
        self.assertTrue(picked[1] in all_cand_sel)
        self.assertEqual(ter.candidate_ratio(), 6./18.)
        self.assertEqual(ter.accurate_ratio(), 2./18.)
        self.assertEqual(ter.failed_ratio(), 10./18.)


    def test_f(self):
        id_f_accu = [ [3, 5, 1], [1, 7, 5] ]
        id_f_cand = [ [4, 7, 6], [8, 6, 0] ]
        id_f_fail = [ [2, 0, 8], [4, 2, 3] ]
        id_v_accu = [None, None]
        id_v_cand = [None, None]
        id_v_fail = [None, None]
        expected_accu = id_f_accu
        expected_cand = id_f_cand
        expected_fail = id_f_fail
        expected_accu = [set(ii) for ii in expected_accu]
        expected_cand = [set(ii) for ii in expected_cand]
        expected_fail = [set(ii) for ii in expected_fail]
        all_cand_sel = [ (0,4), (0,7), (0,6), (1,6), (1,0), (1,8) ]
        
        ter = TrajsExplorationReport()
        for ii in range(2):
            ter.record_traj(
                id_f_accu[ii], id_f_cand[ii], id_f_fail[ii],
                id_v_accu[ii], id_v_cand[ii], id_v_fail[ii],
            )
        self.assertEqual(ter.traj_cand, expected_cand)
        self.assertEqual(ter.traj_accu, expected_accu)
        self.assertEqual(ter.traj_fail, expected_fail)
        
        picked = ter.get_candidates(2)
        self.assertEqual(len(picked), 2)
        self.assertTrue(picked[0] in all_cand_sel)
        self.assertTrue(picked[1] in all_cand_sel)


    def test_f_max(self):
        id_f_accu = [ [3, 5, 1], [1, 7, 5] ]
        id_f_cand = [ [4, 7, 6], [8, 6, 0] ]
        id_f_fail = [ [2, 0, 8], [4, 2, 3] ]
        id_v_accu = [None, None]
        id_v_cand = [None, None]
        id_v_fail = [None, None]
        expected_accu = id_f_accu
        expected_cand = id_f_cand
        expected_fail = id_f_fail
        expected_accu = [set(ii) for ii in expected_accu]
        expected_cand = [set(ii) for ii in expected_cand]
        expected_fail = [set(ii) for ii in expected_fail]
        all_cand_sel = [ (0,4), (0,7), (0,6), (1,6), (1,0), (1,8) ]
        
        ter = TrajsExplorationReport()
        for ii in range(2):
            ter.record_traj(
                id_f_accu[ii], id_f_cand[ii], id_f_fail[ii],
                id_v_accu[ii], id_v_cand[ii], id_v_fail[ii],
            )
        self.assertEqual(ter.traj_cand, expected_cand)
        self.assertEqual(ter.traj_accu, expected_accu)
        self.assertEqual(ter.traj_fail, expected_fail)
        
        picked = ter.get_candidates(10)
        self.assertEqual(len(picked), 6)
        self.assertEqual(
            set(picked),
            set(all_cand_sel),
        )
