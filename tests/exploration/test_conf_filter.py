import unittest

import dpdata
from fake_data_set import (
    fake_system,
)

# isort: off
from dpgen2.exploration.selector import (
    ConfFilter,
    ConfFilters,
)

# isort: on


class FooFilter(ConfFilter):
    def check(
        self,
        frame: dpdata.System,
    ) -> bool:
        return frame["coords"][0][0][0] > 0.0


class BarFilter(ConfFilter):
    def check(
        self,
        frame: dpdata.System,
    ) -> bool:
        return frame["coords"][0][0][1] > 0.0


class BazFilter(ConfFilter):
    def check(
        self,
        frame: dpdata.System,
    ) -> bool:
        return frame["coords"][0][0][2] > 0.0


class TestConfFilter(unittest.TestCase):
    def test_filter_0(self):
        faked_sys = fake_system(4, 3)
        # expected only frame 1 is preseved.
        faked_sys["coords"][1][0] = 1.0
        faked_sys["coords"][0][0][0] = 2.0
        faked_sys["coords"][2][0][1] = 3.0
        faked_sys["coords"][3][0][2] = 4.0
        filters = ConfFilters()
        filters.add(FooFilter()).add(BarFilter()).add(BazFilter())
        ms = dpdata.MultiSystems()
        ms.append(faked_sys)
        sel_sys = filters.check(ms)[0]
        self.assertEqual(sel_sys.get_nframes(), 1)
        self.assertAlmostEqual(sel_sys["coords"][0][0][0], 1)

    def test_filter_1(self):
        faked_sys = fake_system(4, 3)
        # expected frame 1 and 3 are preseved.
        faked_sys["coords"][1][0] = 1.0
        faked_sys["coords"][3][0] = 3.0
        filters = ConfFilters()
        filters.add(FooFilter()).add(BarFilter()).add(BazFilter())
        ms = dpdata.MultiSystems()
        ms.append(faked_sys)
        sel_sys = filters.check(ms)[0]
        self.assertEqual(sel_sys.get_nframes(), 2)
        self.assertAlmostEqual(sel_sys["coords"][0][0][0], 1)
        self.assertAlmostEqual(sel_sys["coords"][1][0][0], 3)

    def test_filter_all(self):
        faked_sys = fake_system(4, 3)
        # expected all frames are preseved.
        faked_sys["coords"][0][0] = 0.5
        faked_sys["coords"][1][0] = 1.0
        faked_sys["coords"][2][0] = 2.0
        faked_sys["coords"][3][0] = 3.0
        filters = ConfFilters()
        filters.add(FooFilter()).add(BarFilter()).add(BazFilter())
        ms = dpdata.MultiSystems()
        ms.append(faked_sys)
        sel_sys = filters.check(ms)[0]
        self.assertEqual(sel_sys.get_nframes(), 4)
        self.assertAlmostEqual(sel_sys["coords"][0][0][0], 0.5)
        self.assertAlmostEqual(sel_sys["coords"][1][0][0], 1)
        self.assertAlmostEqual(sel_sys["coords"][2][0][0], 2)
        self.assertAlmostEqual(sel_sys["coords"][3][0][0], 3)

    def test_filter_none(self):
        faked_sys = fake_system(4, 3)
        filters = ConfFilters()
        filters.add(FooFilter()).add(BarFilter()).add(BazFilter())
        ms = dpdata.MultiSystems()
        ms.append(faked_sys)
        sel_ms = filters.check(ms)
        self.assertEqual(sel_ms.get_nframes(), 0)
