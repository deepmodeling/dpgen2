import json
import os
import random
import re
import shutil
import tempfile
import textwrap
import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
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

# dpdata >= 1.0.2 emits a "Masses" section in the lammps/lmp output
_dpdata_ver = tuple(int(x) for x in re.findall(r"\d+", dpdata.__version__)[:3])
_lmp_masses = (
    "Masses\n\n     1   26.9815386000 # Al\n     2   24.3050000000 # Mg\n\n"
    if _dpdata_ver >= (1, 0, 2)
    else ""
)

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
ofc0 = (
    "\n1 atoms\n2 atom types\n   0.0000000000    2.0000000000 xlo xhi\n   0.0000000000    2.0000000000 ylo yhi\n   0.0000000000    2.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\n"
    + _lmp_masses
    + "Atoms # atomic\n\n     1      1    0.0000000000    0.0000000000    0.0000000000\n"
)

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
ofc1 = (
    "\n1 atoms\n2 atom types\n   0.0000000000    3.0000000000 xlo xhi\n   0.0000000000    3.0000000000 ylo yhi\n   0.0000000000    3.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\n"
    + _lmp_masses
    + "Atoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n"
)

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
ofc2 = (
    "\n1 atoms\n2 atom types\n   0.0000000000    4.0000000000 xlo xhi\n   0.0000000000    4.0000000000 ylo yhi\n   0.0000000000    4.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\n"
    + _lmp_masses
    + "Atoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n"
)

abacus_stru = """ATOMIC_SPECIES
Si 28.085 Si.upf

NUMERICAL_ORBITAL
Si.orb

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0

ATOMIC_POSITIONS
Cartesian

Si
0.0
1
1.0 2.0 3.0 1 1 1 mag 0.0
"""


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
            "remove_spins": False,
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
            "remove_spins": False,
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
    @unittest.skipIf(
        _dpdata_ver < (0, 2, 22), "dpdata does not load/export ABACUS spins"
    )
    def test_abacus_spin_metadata_is_removed_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            stru = Path(tmp) / "STRU"
            stru.write_text(abacus_stru)
            source = dpdata.System(stru, fmt="abacus/stru", type_map=["Si"])
            self.assertIn("spins", source.data)
            generator = FileConfGenerator(
                str(stru), fmt="abacus/stru", remove_spins=True
            )
            with self.assertLogs(level="WARNING") as logs:
                loaded = generator.generate(["Si"])[0]
            self.assertIn("remove_spins", logs.output[0])
            self.assertNotIn("spins", loaded.data)
            for key in ["coords", "cells", "atom_types"]:
                np.testing.assert_allclose(loaded[key], source[key])

            for fmt in ["lammps/lmp", "lmp", "LAMMPS/LMP", "Lmp", "LMP", "lammps/LMP"]:
                with self.subTest(fmt=fmt), self.assertLogs(level="WARNING"):
                    content = generator.get_file_content(type_map=["Si"], fmt=fmt)[0]
                    output = Path(tmp) / "conf.lmp"
                    output.write_text(content)
                    roundtrip = dpdata.System(output, fmt="lammps/lmp", type_map=["Si"])
                    self.assertNotIn("spins", roundtrip.data)
                    for key in ["coords", "cells", "atom_types"]:
                        np.testing.assert_allclose(roundtrip[key], source[key])
            self.assertEqual(stru.read_text(), abacus_stru)
            np.testing.assert_array_equal(
                dpdata.System(stru, fmt="abacus/stru")["spins"], source["spins"]
            )

    @unittest.skipIf(
        _dpdata_ver < (0, 2, 22), "dpdata does not load/export ABACUS spins"
    )
    def test_abacus_spins_are_preserved_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            stru = Path(tmp) / "STRU"
            stru.write_text(abacus_stru.replace("mag 0.0", "mag 2.0"))
            for options in [{}, {"remove_spins": False}]:
                with self.subTest(options=options):
                    generator = FileConfGenerator(
                        str(stru), fmt="abacus/stru", **options
                    )
                    source = generator.generate(["Si"])[0]
                    self.assertIn("spins", source.data)
                    expected = Path(tmp) / "expected.lmp"
                    source[0].to("lammps/lmp", expected)
                    self.assertEqual(
                        generator.get_file_content(["Si"])[0], expected.read_text()
                    )
                    self.assertIn("spins", generator.generate(["Si"])[0].data)

    def test_remove_spins_applies_to_every_mixed_system(self):
        systems = dpdata.MultiSystems(type_map=["Al", "Mg"])
        for poscar in [pos0, pos1]:
            with tempfile.NamedTemporaryFile(mode="w+") as stream:
                stream.write(poscar)
                stream.flush()
                system = dpdata.System(stream.name, fmt="vasp/poscar")
                system.data["spins"] = np.ones_like(system["coords"])
                systems.append(system)
        with patch.object(FileConfGenerator, "generate_mixed", return_value=systems):
            with self.assertLogs(level="WARNING") as logs:
                loaded = FileConfGenerator(
                    "unused", fmt="deepmd/npy/mixed", remove_spins=True
                ).generate(["Al", "Mg"])
        self.assertEqual(len(loaded), 2)
        self.assertEqual(len(logs.output), 2)
        for system in loaded:
            self.assertNotIn("spins", system.data)

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
