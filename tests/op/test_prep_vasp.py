from context import dpgen2
import os,sys,json,glob,shutil,textwrap
import dpdata
import numpy as np
import unittest
from dpgen2.fp.vasp import VaspInputs
from dpgen2.op.prep_vasp import PrepVasp
from pathlib import Path
from dpgen2.constants import (
    vasp_task_pattern,
    vasp_input_name,
    vasp_pot_name,
    vasp_kp_name,
    vasp_conf_name,
)
from fake_data_set import fake_system, fake_multi_sys
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
    FatalError,
)


class TestPrepVasp(unittest.TestCase):
    def setUp(self):
        Path('template.incar').write_text('foo')
        Path('POTCAR_H').write_text('bar H\n')
        Path('POTCAR_O').write_text('bar O\n')
        self.nframes_0 = [2, 5, 3]
        self.natoms_0 = [4, 3, 4]
        self.nframes_1 = [3, 4, 2]
        self.natoms_1 = [5, 3, 2]
        ms_0 = fake_multi_sys( self.nframes_0, self.natoms_0, 'O')
        ms_1 = fake_multi_sys( self.nframes_1, self.natoms_1, 'H')
        ms_0.to_deepmd_npy('data-0')
        ms_1.to_deepmd_npy('data-1')
        self.confs = ['data-0', 'data-1']
        self.confs = [Path(ii) for ii in self.confs]    
        
    def tearDown(self):
        os.remove('template.incar')
        os.remove('POTCAR_H')
        os.remove('POTCAR_O')
        for ii in self.confs:
            if ii.is_dir():
                shutil.rmtree(ii)
        tot_f_0 = sum(self.nframes_0)
        tot_f_1 = sum(self.nframes_1)
        tot_f = tot_f_0 + tot_f_1
        for ii in range(tot_f):
            tname = Path(vasp_task_pattern%ii)
            if tname.is_dir():
                shutil.rmtree(tname)
        
    def check_sys(self, ss0, ss1):
        self.assertEqual(ss0['atom_numbs'], ss1['atom_numbs'])
        self.assertEqual(ss0['atom_names'], ss1['atom_names'])

        
    def test(self):
        refkp = textwrap.dedent(
"""Automatic mesh
0
Gamma
63 63 63
0  0  0
""")
        iincar = 'template.incar'
        ipotcar = {'H' : 'POTCAR_H', 'O' : 'POTCAR_O'}
        vi = VaspInputs(0.1, True, iincar, ipotcar)
        op = PrepVasp()
        opout = op.execute(OPIO({
            'inputs': vi,
            'confs' : self.confs,
        }))
        task_names = opout['task_names']
        task_paths = opout['task_paths']
        tot_f_0 = sum(self.nframes_0)
        tot_f_1 = sum(self.nframes_1)
        tot_f = tot_f_0 + tot_f_1
        for ii in range(tot_f):
            self.assertEqual(task_names[ii], vasp_task_pattern%ii)
            self.assertEqual(str(task_paths[ii]), vasp_task_pattern%ii)
        for ii in range(tot_f_0):
            ipath = task_paths[ii]
            self.assertEqual('foo', (ipath/vasp_input_name).read_text())
            self.assertEqual('bar O\n', (ipath/vasp_pot_name).read_text())
            self.assertEqual(refkp, (ipath/vasp_kp_name).read_text())
        for ii in range(tot_f_0, tot_f):
            ipath = task_paths[ii]
            self.assertEqual('foo', (ipath/vasp_input_name).read_text())
            self.assertEqual('bar H\n', (ipath/vasp_pot_name).read_text())
            self.assertEqual(refkp, (ipath/vasp_kp_name).read_text())
        ms0 = dpdata.MultiSystems()
        ms0.from_deepmd_npy(self.confs[0])
        ms1 = dpdata.MultiSystems()
        ms1.from_deepmd_npy(self.confs[1])
        self.check_sys(dpdata.System(task_paths[0]/vasp_conf_name), ms0[0][0])
        self.check_sys(dpdata.System(task_paths[1]/vasp_conf_name), ms0[0][1])
        self.check_sys(dpdata.System(task_paths[2]/vasp_conf_name), ms0[0][2])
        self.check_sys(dpdata.System(task_paths[3]/vasp_conf_name), ms0[0][3])
        self.check_sys(dpdata.System(task_paths[4]/vasp_conf_name), ms0[0][4])
        self.check_sys(dpdata.System(task_paths[5]/vasp_conf_name), ms0[1][0])
        self.check_sys(dpdata.System(task_paths[6]/vasp_conf_name), ms0[1][1])
        self.check_sys(dpdata.System(task_paths[7]/vasp_conf_name), ms0[1][2])
        self.check_sys(dpdata.System(task_paths[8]/vasp_conf_name), ms0[1][3])
        self.check_sys(dpdata.System(task_paths[9]/vasp_conf_name), ms0[1][4])
        self.check_sys(dpdata.System(task_paths[10]/vasp_conf_name), ms1[0][0])
        self.check_sys(dpdata.System(task_paths[11]/vasp_conf_name), ms1[0][1])
        self.check_sys(dpdata.System(task_paths[12]/vasp_conf_name), ms1[1][0])
        self.check_sys(dpdata.System(task_paths[13]/vasp_conf_name), ms1[1][1])
        self.check_sys(dpdata.System(task_paths[14]/vasp_conf_name), ms1[1][2])
        self.check_sys(dpdata.System(task_paths[15]/vasp_conf_name), ms1[1][3])
        self.check_sys(dpdata.System(task_paths[16]/vasp_conf_name), ms1[2][0])
        self.check_sys(dpdata.System(task_paths[17]/vasp_conf_name), ms1[2][1])
        self.check_sys(dpdata.System(task_paths[18]/vasp_conf_name), ms1[2][2])
