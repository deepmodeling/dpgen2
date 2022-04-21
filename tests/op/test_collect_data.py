import dpdata
from op.context import dpgen2
import numpy as np
import unittest, json, shutil
from mock import mock, patch, call
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
)
from pathlib import Path
from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_traj_name,
    lmp_model_devi_name,
)
from dpgen2.op.collect_data import CollectData
from fake_data_set import fake_system, fake_multi_sys

class TestRunLmp(unittest.TestCase):
    def setUp(self):
        self.iter_data = [Path('iter0')]
        self.labeled_data = [Path('data0'), Path('data1')]    
        # fake multi sys
        self.atom_name_ms = 'foo'
        self.natoms = [1, 2]
        self.nframes = [3, 4]
        ms = fake_multi_sys(self.nframes, self.natoms, self.atom_name_ms)
        # fake sys
        self.atom_name = 'bar'
        self.natoms_0 = 3
        self.natoms_1 = 4
        self.nframes_0 = 6
        self.nframes_1 = 5
        ss_0 = fake_system(self.nframes_0, self.natoms_0, self.atom_name)
        ss_1 = fake_system(self.nframes_1, self.natoms_1, self.atom_name)        
        # dump
        ms.to_deepmd_npy(self.iter_data[0])
        ss_0.to_deepmd_npy(self.labeled_data[0])
        ss_1.to_deepmd_npy(self.labeled_data[1])

    def tearDown(self):
        for ii in ['iter0', 'iter1']:
            if Path(ii).is_dir():
                shutil.rmtree(ii)        
        for ii in ['data0', 'data1']:
            if Path(ii).is_dir():
                shutil.rmtree(ii)        

    def test_success(self):
        op = CollectData()
        out = op.execute(
            OPIO({
                'name' : 'iter1',
                'iter_data' : self.iter_data,
                'labeled_data' : self.labeled_data,
            }))
        self.assertTrue(Path('iter1').is_dir())
        self.assertEqual(out['iter_data'], [Path('iter0'), Path('iter1')])
        ms = dpdata.MultiSystems()
        ms.from_deepmd_npy(out['iter_data'][1])
        self.assertEqual(sorted(ms.systems.keys()), ['bar3', 'bar4'])
        self.assertEqual(ms.systems['bar3'].get_nframes(), 6)
        self.assertEqual(ms.systems['bar4'].get_nframes(), 5)
        
