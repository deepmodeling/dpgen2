from utils.context import dpgen2
import numpy as np
import unittest, json, shutil, os
import random
import tempfile
import dpdata
from pathlib import Path
from dpgen2.entrypoint.submit import (
    make_conf_list,
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
ofc0 = '\n1 atoms\n2 atom types\n   0.0000000000    2.0000000000 xlo xhi\n   0.0000000000    2.0000000000 ylo yhi\n   0.0000000000    2.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      1    0.0000000000    0.0000000000    0.0000000000\n'

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
ofc1 = '\n1 atoms\n2 atom types\n   0.0000000000    3.0000000000 xlo xhi\n   0.0000000000    3.0000000000 ylo yhi\n   0.0000000000    3.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n'

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
ofc2 = '\n1 atoms\n2 atom types\n   0.0000000000    4.0000000000 xlo xhi\n   0.0000000000    4.0000000000 ylo yhi\n   0.0000000000    4.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n'


class TestSubmit(unittest.TestCase):
    def test_make_conf_list_path_list(self):
        f0 = Path('f0.POSCAR')
        f1 = Path('f1.POSCAR')
        f2 = Path('d0.POSCAR')
        f0.write_text(ifc0)
        f1.write_text(ifc1)
        f2.write_text(ifc2)
        ret = make_conf_list(['f*POSCAR', 'd0.POSCAR'], type_map=['Al', 'Mg'])
        f0.unlink()
        f1.unlink()
        f2.unlink()
        self.assertEqual(ret, [ofc0, ofc1, ofc2])

    def test_make_conf_list_alloy_conf(self):
        idict = {
            'lattice' : ('sc', 3.),
            'numb_confs' : 3,
            'concentration' : [0., 1.],
        }
        ret = make_conf_list(idict, type_map=['Al', 'Mg'])
        self.assertEqual(ret, [ofc1, ofc1, ofc1])
