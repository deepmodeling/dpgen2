"""Check explicit fixture regeneration without changing committed data."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import (
    Path,
)

import dpdata
import numpy as np

from dpgen2.fp.vasp import (
    make_kspacing_kpoints,
)

from .context import (
    dpgen2,
)


class TestMakeKpData(unittest.TestCase):
    def test_generator_writes_matching_references_from_other_directory(self):
        source = Path(__file__).parent / "data.vasp.kp.gf"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            for name in ("make_kp_data.py", "POSCAR"):
                shutil.copy(source / name, fixtures / name)
            subprocess.run(
                [sys.executable, str(fixtures / "make_kp_data.py")],
                cwd=root,
                check=True,
            )
            generated = sorted(fixtures.glob("test.*"))
            self.assertEqual(len(generated), 30)
            for fixture in generated:
                cell = dpdata.System(fixture / "POSCAR")["cells"][0]
                expected = make_kspacing_kpoints(cell, 0.16, False).splitlines()[3]
                np.testing.assert_array_equal(
                    np.loadtxt(fixture / "kp.ref", dtype=int),
                    np.array(expected.split(), dtype=int),
                )
