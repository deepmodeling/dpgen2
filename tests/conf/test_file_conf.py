import json
import os
import random
import shutil
import tempfile
import textwrap
import unittest
from pathlib import (
    Path,
)

import dpdata
import numpy as np

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.conf.file_conf import (
    FileConfGenerator,
)

# isort: on

pos0 = textwrap.dedent(
    """POSCAR file written by OVITO
1
1 0 0 
0 1 0 
0 0 1
Al 
1
Cartesian
0 0 0 
"""
)

pos1 = textwrap.dedent(
    """POSCAR file written by OVITO
1
2 0 0 
0 2 0
0 0 2
Mg
1
Cartesian
0 0 0 
"""
)

ifc0 = """Al1 
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Al 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc0 = "\n1 atoms\n2 atom types\n   0.0000000000    2.0000000000 xlo xhi\n   0.0000000000    2.0000000000 ylo yhi\n   0.0000000000    2.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      1    0.0000000000    0.0000000000    0.0000000000\n"

ifc1 = """Mg1 
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Mg 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc1 = "\n1 atoms\n2 atom types\n   0.0000000000    3.0000000000 xlo xhi\n   0.0000000000    3.0000000000 ylo yhi\n   0.0000000000    3.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n"

ifc2 = """Mg1 
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Mg 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc2 = "\n1 atoms\n2 atom types\n   0.0000000000    4.0000000000 xlo xhi\n   0.0000000000    4.0000000000 ylo yhi\n   0.0000000000    4.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n"


class TestFileConfGenerator(unittest.TestCase):
    def setUp(self):
        self.prefix = "mytmp___"
        Path(self.prefix).mkdir(exist_ok=True)
        (Path(self.prefix) / Path("poscar.foo.0")).write_text(pos0)
        (Path(self.prefix) / Path("poscar.foo.1")).write_text(pos1)

    def tearDown(self):
        if Path(self.prefix).is_dir():
            shutil.rmtree(self.prefix)

    def test_list(self):
        fcg = FileConfGenerator(
            ["poscar.foo.0", "poscar.foo.1"],
            fmt="vasp/poscar",
            prefix=self.prefix,
        )
        ms = fcg.generate(["Cu", "Al", "Mg"])
        self.assertEqual(len(ms), 2)
        self.assertEqual(ms[0]["atom_names"], ["Cu", "Al", "Mg"])
        self.assertEqual(ms[0]["atom_numbs"], [0, 1, 0])
        self.assertEqual(ms[0]["atom_types"], [1])
        self.assertAlmostEqual(ms[0]["cells"][0][0][0], 1.0)
        self.assertEqual(ms[1]["atom_names"], ["Cu", "Al", "Mg"])
        self.assertEqual(ms[1]["atom_numbs"], [0, 0, 1])
        self.assertEqual(ms[1]["atom_types"], [2])
        self.assertAlmostEqual(ms[1]["cells"][0][0][0], 2.0)

    def test_widecard(self):
        fcg = FileConfGenerator(
            "poscar.foo.*",
            fmt="vasp/poscar",
            prefix=self.prefix,
        )
        ms = fcg.generate(["Cu", "Al", "Mg"])
        self.assertEqual(len(ms), 2)
        self.assertEqual(ms[0]["atom_names"], ["Cu", "Al", "Mg"])
        self.assertEqual(ms[0]["atom_numbs"], [0, 1, 0])
        self.assertEqual(ms[0]["atom_types"], [1])
        self.assertAlmostEqual(ms[0]["cells"][0][0][0], 1.0)
        self.assertEqual(ms[1]["atom_names"], ["Cu", "Al", "Mg"])
        self.assertEqual(ms[1]["atom_numbs"], [0, 0, 1])
        self.assertEqual(ms[1]["atom_types"], [2])
        self.assertAlmostEqual(ms[1]["cells"][0][0][0], 2.0)

    def test_normalize(self):
        in_data = {
            "files": "foo",
        }
        expected_out_data = {
            "files": "foo",
            "fmt": "auto",
            "prefix": None,
            "remove_pbc": False,
        }
        out_data = FileConfGenerator.normalize_config(
            in_data,
        )
        self.assertEqual(out_data, expected_out_data)

    def test_normalize_1(self):
        in_data = {
            "files": ["foo"],
            "fmt": "bar",
        }
        expected_out_data = {
            "files": ["foo"],
            "fmt": "bar",
            "prefix": None,
            "remove_pbc": False,
        }
        out_data = FileConfGenerator.normalize_config(
            in_data,
        )
        self.assertEqual(out_data, expected_out_data)

    def test_deepmd_mixed(self):
        type_map = ["Cu", "Al", "Mg"]
        ms = dpdata.MultiSystems(type_map=type_map)
        ms.append(
            dpdata.System(Path(self.prefix) / Path("poscar.foo.0"), fmt="vasp/poscar")
        )
        ms.append(
            dpdata.System(Path(self.prefix) / Path("poscar.foo.1"), fmt="vasp/poscar")
        )
        ms.to("deepmd/npy/mixed", "test_mixed")
        fcg = FileConfGenerator("test_mixed", fmt="deepmd/npy/mixed")
        ms1 = fcg.generate(type_map)
        self.assertEqual(len(ms), len(ms1))
        for tt in ["atom_names", "atom_numbs", "atom_types"]:
            self.assertEqual(ms[0][tt], ms1[0][tt])
            self.assertEqual(ms[1][tt], ms1[1][tt])
        for tt in ["cells", "coords"]:
            np.testing.assert_almost_equal(ms[0][tt], ms1[0][tt])
            np.testing.assert_almost_equal(ms[1][tt], ms1[1][tt])
        shutil.rmtree("test_mixed")


class TestFileConfGeneratorContent(unittest.TestCase):
    def test_list_1(self):
        f0 = Path("f0.POSCAR")
        f1 = Path("f1.POSCAR")
        f2 = Path("d0.POSCAR")
        f0.write_text(ifc0)
        f1.write_text(ifc1)
        f2.write_text(ifc2)
        fcg = FileConfGenerator(["f*POSCAR", "d0.POSCAR"])
        ret = fcg.get_file_content(type_map=["Al", "Mg"])
        f0.unlink()
        f1.unlink()
        f2.unlink()
        self.assertEqual(ret, [ofc0, ofc1, ofc2])
