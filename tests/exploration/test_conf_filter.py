import os
import numpy as np
import unittest, dpdata
from .context import dpgen2
from dpgen2.exploration.selector import ConfFilter, ConfFilters
from fake_data_set import fake_system
from mock import patch

class FooFilter(ConfFilter):
    def check (
            self,
            coords : np.array,
            cell: np.array,
            atom_types : np.array,
            nopbc: bool,
    ) -> bool :
        return True


class faked_filter():
    myiter = -1
    myret = [True]
    @classmethod
    def faked_check(
        cls, cc, ce, at, np):
        cls.myiter += 1
        cls.myiter = cls.myiter % len(cls.myret)
        return cls.myret[cls.myiter]


class TestConfFilter(unittest.TestCase):
    @patch.object(FooFilter, "check", faked_filter.faked_check)
    def test_filter_0(self):
        faked_filter.myiter = -1
        faked_filter.myret = [
            True, True, False, True,
            False, True, True, False,
            True, True, False, False,
        ]
        faked_sys = fake_system(4, 3)
        # expected only frame 1 is preseved.
        faked_sys['coords'][1][0][0] = 1.
        filters = ConfFilters()
        filters.add(FooFilter()).add(FooFilter()).add(FooFilter())
        sel_sys = filters.check(faked_sys)
        self.assertEqual(sel_sys.get_nframes(), 1)
        self.assertAlmostEqual(sel_sys['coords'][0][0][0], 1)

    @patch.object(FooFilter, "check", faked_filter.faked_check)
    def test_filter_1(self):
        faked_filter.myiter = -1
        faked_filter.myret = [
            True, True, False, True,
            False, True, True, True,
            True, True, False, True,
        ]
        faked_sys = fake_system(4, 3)
        # expected frame 1 and 3 are preseved.
        faked_sys['coords'][1][0][0] = 1.
        faked_sys['coords'][3][0][0] = 3.
        filters = ConfFilters()
        filters.add(FooFilter()).add(FooFilter()).add(FooFilter())
        sel_sys = filters.check(faked_sys)
        self.assertEqual(sel_sys.get_nframes(), 2)
        self.assertAlmostEqual(sel_sys['coords'][0][0][0], 1)
        self.assertAlmostEqual(sel_sys['coords'][1][0][0], 3)

    @patch.object(FooFilter, "check", faked_filter.faked_check)
    def test_filter_all(self):
        faked_filter.myiter = -1
        faked_filter.myret = [
            True, True, True, True,
        ]
        faked_sys = fake_system(4, 3)
        # expected all frames are preseved.
        faked_sys['coords'][0][0][0] = .5
        faked_sys['coords'][1][0][0] = 1.
        faked_sys['coords'][2][0][0] = 2.
        faked_sys['coords'][3][0][0] = 3.
        filters = ConfFilters()
        filters.add(FooFilter()).add(FooFilter()).add(FooFilter())
        sel_sys = filters.check(faked_sys)
        self.assertEqual(sel_sys.get_nframes(), 4)
        self.assertAlmostEqual(sel_sys['coords'][0][0][0], .5)
        self.assertAlmostEqual(sel_sys['coords'][1][0][0], 1)
        self.assertAlmostEqual(sel_sys['coords'][2][0][0], 2)
        self.assertAlmostEqual(sel_sys['coords'][3][0][0], 3)

    @patch.object(FooFilter, "check", faked_filter.faked_check)
    def test_filter_none(self):
        faked_filter.myiter = -1
        faked_filter.myret = [
            False, False, False, False,
        ]
        faked_sys = fake_system(4, 3)
        filters = ConfFilters()
        filters.add(FooFilter()).add(FooFilter()).add(FooFilter())
        sel_sys = filters.check(faked_sys)
        self.assertEqual(sel_sys.get_nframes(), 0)
        
