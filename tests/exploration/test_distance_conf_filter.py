import os
import unittest

import dpdata

from dpgen2.exploration.selector import (
    BoxLengthFilter,
    BoxSkewnessConfFilter,
    DistanceConfFilter,
)


POSCAR_valid = """ Er
1.0
   7.00390434172054       0.000000000000000E+000  0.000000000000000E+000
  -3.50195193887670        6.06555921954188       0.000000000000000E+000
  4.695904554609645E-007  8.133544916878595E-007   6.21991417332993
 Er    Th       Pd
           3           3           5
Direct
  0.404315576593774       0.000000000000000E+000  0.916328931175151
  0.000000000000000E+000  0.404315576593774       0.916328931175151
  0.595684423406226       0.595684423406226       0.916328931175151
  0.308657693786501       0.000000000000000E+000  0.431543321200265
  0.000000000000000E+000  0.308657693786501       0.431543321200265
  0.691342306213499       0.691342306213499       0.431543321200265
  0.333299994468689       0.666700005531311       0.181639126706554
  0.666700005531311       0.333299994468689       0.181639126706554
  0.333299994468689       0.666700005531311       0.653715146968972
  0.666700005531311       0.333299994468689       0.653715146968972
  0.000000000000000E+000  0.000000000000000E+000  0.767627989523288
"""


POSCAR_tilt = """POSCAR file written by OVITO Basic 3.10.6
1
9.9156076829 0.0 0.0
7.9377192882e-07 10.5138279814 0.0
11.9805 1.9108776119e-08 6.3054328214
Re B Au
3 2 4
Direct
0.4552610161 0.1437969637 0.9105503417
0.4552610161 0.8562030363 0.9105503417
0.9406309225 4.4367368569e-14 0.8634531812
0.1522944187 0.5 0.9114412858
0.5212391304 -3.0871058636e-12 0.1011293077
0.8742903123 0.6505092137 0.9508321387
0.8742903123 0.3494907863 0.9508321387
0.5509009668 0.6521996574 0.004767873
0.5509009668 0.3478003426 0.004767873
"""


POSCAR_long = """POSCAR file written by OVITO Basic 3.10.6
1
11.8987292195 0.0 0.0
9.5252631458e-07 12.6165935777 0.0
0.0 2.2930531343e-08 2.0177385028
Re B Au
3 2 4
Direct
0.4552610161 0.1437969637 0.9105503417
0.4552610161 0.8562030363 0.9105503417
0.9406309225 4.4367368568e-14 0.8634531812
0.1522944187 0.5 0.9114412858
0.5212391304 -3.0871058636e-12 0.1011293077
0.8742903123 0.6505092137 0.9508321387
0.8742903123 0.3494907863 0.9508321387
0.5509009668 0.6521996574 0.004767873
0.5509009668 0.3478003426 0.004767873
"""


POSCAR_close = """POSCAR file written by OVITO Basic 3.10.6
1
9.9156076829 0.0 0.0
7.9377192882e-07 10.5138279814 0.0
3.0151272179 1.9108776119e-08 6.3054328214
Re B Au
3 2 4
Direct
0.4552610161 0.1437969637 0.9105503417
0.5078872031 0.9988722905 1.069143737
0.9406309225 4.4367368569e-14 0.8634531812
0.1522944187 0.5 0.9114412858
0.5212391304 -3.0871058636e-12 0.1011293077
0.8742903123 0.6505092137 0.9508321387
0.8742903123 0.3494907863 0.9508321387
0.5509009668 0.6521996574 0.004767873
0.5509009668 0.3478003426 0.004767873
"""


class TestBoxSkewnessConfFilter(unittest.TestCase):
    def setUp(self):
        with open("POSCAR_valid", "w") as f:
            f.write(POSCAR_valid)
        with open("POSCAR_tilt", "w") as f:
            f.write(POSCAR_tilt)

    def test_valid(self):
        system = dpdata.System("POSCAR_valid", fmt="poscar")
        distance_conf_filter = BoxSkewnessConfFilter()
        valid = distance_conf_filter.check(system)
        self.assertTrue(valid)

    def test_invalid(self):
        system = dpdata.System("POSCAR_tilt", fmt="poscar")
        distance_conf_filter = BoxSkewnessConfFilter()
        valid = distance_conf_filter.check(system)
        self.assertFalse(valid)

    def tearDown(self):
        if os.path.isfile("POSCAR_valid"):
            os.remove("POSCAR_valid")
        if os.path.isfile("POSCAR_tilt"):
            os.remove("POSCAR_tilt")


class TestBoxLengthConfFilter(unittest.TestCase):
    def setUp(self):
        with open("POSCAR_valid", "w") as f:
            f.write(POSCAR_valid)
        with open("POSCAR_long", "w") as f:
            f.write(POSCAR_long)

    def test_valid(self):
        system = dpdata.System("POSCAR_valid", fmt="poscar")
        distance_conf_filter = BoxLengthFilter()
        valid = distance_conf_filter.check(system)
        self.assertTrue(valid)

    def test_invalid(self):
        system = dpdata.System("POSCAR_long", fmt="poscar")
        distance_conf_filter = BoxLengthFilter()
        valid = distance_conf_filter.check(system)
        self.assertFalse(valid)

    def tearDown(self):
        if os.path.isfile("POSCAR_valid"):
            os.remove("POSCAR_valid")
        if os.path.isfile("POSCAR_long"):
            os.remove("POSCAR_long")


class TestDistanceConfFilter(unittest.TestCase):
    def setUp(self):
        with open("POSCAR_valid", "w") as f:
            f.write(POSCAR_valid)
        with open("POSCAR_close", "w") as f:
            f.write(POSCAR_close)

    def test_valid(self):
        system = dpdata.System("POSCAR_valid", fmt="poscar")
        distance_conf_filter = DistanceConfFilter()
        valid = distance_conf_filter.check(system)
        self.assertTrue(valid)

    def test_invalid(self):
        system = dpdata.System("POSCAR_close", fmt="poscar")
        distance_conf_filter = DistanceConfFilter()
        valid = distance_conf_filter.check(system)
        self.assertFalse(valid)

    def tearDown(self):
        if os.path.isfile("POSCAR_valid"):
            os.remove("POSCAR_valid")
        if os.path.isfile("POSCAR_close"):
            os.remove("POSCAR_close")
