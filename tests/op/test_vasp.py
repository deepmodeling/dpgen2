from context import dpgen2
import os,sys,json,glob,shutil,textwrap
import dpdata
import numpy as np
import unittest
from dpgen2.fp.vasp import make_kspacing_kpoints, VaspInputs
from pathlib import Path

class TestVASPInputs(unittest.TestCase):
    def setUp(self):
        Path('template.incar').write_text('foo')
        Path('POTCAR_H').write_text('bar H\n')
        Path('POTCAR_O').write_text('bar O\n')

    def tearDown(self):
        os.remove('template.incar')
        os.remove('POTCAR_H')
        os.remove('POTCAR_O')        
        if Path('POSCAR').is_file():
            os.remove('POSCAR')

    def test_make_kp (self):
        kspacing = 0.16
        gamma = False
        test_path = (Path(__file__).parent)
        all_test = glob.glob(os.path.join(test_path/'data.vasp.kp.gf', 'test.*'))
        self.assertEqual(len(all_test), 30)
        for ii in all_test :
            ss = dpdata.System(Path(ii)/'POSCAR')            
            ret=make_kspacing_kpoints(ss['cells'][0], kspacing, gamma)
            kp = [int(jj) for jj in (ret.split('\n')[3].split())]
            kp_ref = list(np.loadtxt(os.path.join(ii, 'kp.ref'), dtype = int))
            self.assertTrue(kp == kp_ref)

    def test_vasp_input_incar_potcar(self):
        iincar = 'template.incar'
        ipotcar = {'H' : 'POTCAR_H', 'O' : 'POTCAR_O'}
        vi = VaspInputs(0.16, True, iincar, ipotcar)
        self.assertEqual(vi.incar_template, 'foo')
        self.assertEqual(vi.potcars['O'], 'bar O\n')
        self.assertEqual(vi.potcars['H'], 'bar H\n')
        atom_names = ['O', 'H']
        self.assertEqual(vi.make_potcar(atom_names), 'bar O\nbar H\n')

    def test_vasp_input_kp(self):
        ref = textwrap.dedent(
"""Automatic mesh
0
Gamma
10 7 7
0  0  0
"""
        )
        poscar = textwrap.dedent(
"""Foo
1
0.00 6.00 6.00
8.00 0.00 8.00
9.00 9.00 0.00
O 
1 
Selective dynamics
Cartesian
0.00 0.00 0.00 T T F
"""
        )
        Path('POSCAR').write_text(poscar)
        iincar = 'template.incar'
        ipotcar = {'H' : 'POTCAR_H', 'O' : 'POTCAR_O'}
        vi = VaspInputs(0.1, True, iincar, ipotcar)
        ss = dpdata.System('POSCAR')
        kps = vi.make_kpoints(ss['cells'][0])
        self.assertEqual(ref, kps)

        
    def test_vasp_input_kp(self):
        ref = textwrap.dedent(
"""K-Points
0
Monkhorst Pack
10 7 7
0  0  0
"""
        )
        poscar = textwrap.dedent(
"""Cubic BN
1
0.00 6.00 6.00
8.00 0.00 8.00
9.00 9.00 0.00
O 
1 
Selective dynamics
Cartesian
0.00 0.00 0.00 T T F
"""
        )
        Path('POSCAR').write_text(poscar)
        iincar = 'template.incar'
        ipotcar = {'H' : 'POTCAR_H', 'O' : 'POTCAR_O'}
        vi = VaspInputs(0.1, False, iincar, ipotcar)
        ss = dpdata.System('POSCAR')
        kps = vi.make_kpoints(ss['cells'][0])
        self.assertEqual(ref, kps)

        
