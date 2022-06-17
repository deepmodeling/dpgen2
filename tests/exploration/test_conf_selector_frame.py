from context import dpgen2
import os, textwrap, dpdata, shutil
import numpy as np
import unittest
from pathlib import Path
from dpgen2.exploration.selector import (
    TrustLevel,
    ConfSelectorLammpsFrames,
)

class TestConfSelectorLammpsFrames(unittest.TestCase):
    def setUp(self):
        self.dump_file = textwrap.dedent(
            """ITEM: TIMESTEP
            0
            ITEM: NUMBER OF ATOMS
            3
            ITEM: BOX BOUNDS xy xz yz pp pp pp
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            ITEM: ATOMS id type x y z fx fy fz 
            1 2 11.09 2.87 2.74 0.183043 -0.287677 -0.0974527 
            2 1 11.83 2.56 2.18 -0.224674 0.5841 0.074659 
            3 2 12.25 3.32 1.68 0.0416311 -0.296424 0.0227936 
            ITEM: TIMESTEP
            1
            ITEM: NUMBER OF ATOMS
            3
            ITEM: BOX BOUNDS xy xz yz pp pp pp
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            ITEM: ATOMS id type x y z fx fy fz 
            1 2 11.09 3.87 2.74 0.183043 -0.287677 -0.0974527 
            2 1 11.83 3.56 2.18 -0.224674 0.5841 0.074659 
            3 2 12.25 4.32 1.68 0.0416311 -0.296424 0.0227936 
            ITEM: TIMESTEP
            1
            ITEM: NUMBER OF ATOMS
            3
            ITEM: BOX BOUNDS xy xz yz pp pp pp
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            0.0000000000000000e+00 1.2444699999999999e+01 0.0000000000000000e+00
            ITEM: ATOMS id type x y z fx fy fz 
            1 2 11.09 4.87 2.74 0.183043 -0.287677 -0.0974527 
            2 1 11.83 4.56 2.18 -0.224674 0.5841 0.074659 
            3 2 12.25 5.32 1.68 0.0416311 -0.296424 0.0227936 
            """)
        self.model_devi_file = textwrap.dedent(
            """ #
            0 0.1 0.0 0.0 0.2 0.0 0.0
            0 0.2 0.0 0.0 0.3 0.0 0.0
            0 0.3 0.0 0.0 0.4 0.0 0.0
            """)
        
        self.trajs = [Path('foo.dump'), Path('bar.dump')]
        self.model_devis = [Path('foo.md'), Path('bar.md')]
        for ii in self.trajs:
            ii.write_text(self.dump_file)
        for ii in self.model_devis:
            ii.write_text(self.model_devi_file)
        
        self.traj_fmt = 'lammps/dump'
        self.type_map = ['O', 'H']

    def tearDown(self):
        for ii in ['foo.dump', 'bar.dump', 'foo.md', 'bar.md']:
            if Path(ii).is_file():
                os.remove(ii)
        for ii in ['confs']:
            if Path(ii).is_dir():
                shutil.rmtree(ii)

    def test_f_0(self):
        conf_selector = ConfSelectorLammpsFrames(
            TrustLevel(0.1, 0.5),
        )
        confs, report = conf_selector.select(
            self.trajs, self.model_devis, self.traj_fmt, self.type_map)
        ms = dpdata.MultiSystems(type_map=self.type_map)
        ms.from_deepmd_npy(confs[0], labeled=False)
        self.assertEqual(len(ms), 1)
        ss = ms[0]
        self.assertEqual(ss.get_nframes(), 6)
        self.assertAlmostEqual(ss['coords'][0][0][1], 2.87, places=2)
        self.assertAlmostEqual(ss['coords'][1][0][1], 3.87, places=2)
        self.assertAlmostEqual(ss['coords'][2][0][1], 4.87, places=2)
        self.assertAlmostEqual(ss['coords'][3][0][1], 2.87, places=2)
        self.assertAlmostEqual(ss['coords'][4][0][1], 3.87, places=2)
        self.assertAlmostEqual(ss['coords'][5][0][1], 4.87, places=2)
        # self.assertAlmostEqual(report.ratio('force', 'candidate'), 1.)
        # self.assertAlmostEqual(report.ratio('force', 'accurate'), 0.)
        # self.assertAlmostEqual(report.ratio('force', 'failed'), 0.)
        self.assertAlmostEqual(report.candidate_ratio(), 1.)
        self.assertAlmostEqual(report.accurate_ratio(), 0.)
        self.assertAlmostEqual(report.failed_ratio(), 0.)

    def test_f_1(self):
        conf_selector = ConfSelectorLammpsFrames(
            TrustLevel(0.25, 0.35),
        )
        confs, report = conf_selector.select(
            self.trajs, self.model_devis, self.traj_fmt, self.type_map)
        ms = dpdata.MultiSystems(type_map=self.type_map)
        ms.from_deepmd_npy(confs[0], labeled=False)
        self.assertEqual(len(ms), 1)
        ss = ms[0]
        self.assertEqual(ss.get_nframes(), 2)
        self.assertAlmostEqual(ss['coords'][0][0][1], 3.87, places=2)
        self.assertAlmostEqual(ss['coords'][1][0][1], 3.87, places=2)
        # self.assertAlmostEqual(report.ratio('force', 'candidate'), 1./3.)
        # self.assertAlmostEqual(report.ratio('force', 'accurate'), 1./3.)
        # self.assertAlmostEqual(report.ratio('force', 'failed'), 1./3.)
        self.assertAlmostEqual(report.candidate_ratio(), 1./3.)
        self.assertAlmostEqual(report.accurate_ratio(), 1./3.)
        self.assertAlmostEqual(report.failed_ratio(), 1./3.)
        

    def test_fv_0(self):
        conf_selector = ConfSelectorLammpsFrames(
            TrustLevel(0.25, 0.35, 0.05, 0.15),
        )
        confs, report = conf_selector.select(
            self.trajs, self.model_devis, self.traj_fmt, self.type_map)
        ms = dpdata.MultiSystems(type_map=self.type_map)
        ms.from_deepmd_npy(confs[0], labeled=False)
        self.assertEqual(len(ms), 1)
        ss = ms[0]
        self.assertEqual(ss.get_nframes(), 2)
        self.assertAlmostEqual(ss['coords'][0][0][1], 2.87, places=2)
        self.assertAlmostEqual(ss['coords'][1][0][1], 2.87, places=2)
        # self.assertAlmostEqual(report.ratio('force', 'candidate'), 1./3.)
        # self.assertAlmostEqual(report.ratio('force', 'accurate'), 1./3.)
        # self.assertAlmostEqual(report.ratio('force', 'failed'), 1./3.)
        # self.assertAlmostEqual(report.ratio('virial', 'candidate'), 1./3.)
        # self.assertAlmostEqual(report.ratio('virial', 'accurate'), 0.)
        # self.assertAlmostEqual(report.ratio('virial', 'failed'), 2./3.)
        self.assertAlmostEqual(report.candidate_ratio(), 1./3.)
        self.assertAlmostEqual(report.accurate_ratio(), 0./3.)
        self.assertAlmostEqual(report.failed_ratio(), 2./3.)
        
    def test_fv_1(self):
        conf_selector = ConfSelectorLammpsFrames(
            TrustLevel(0.25, 0.35, 0.05, 0.15),
            max_numb_sel = 1,
        )
        confs, report = conf_selector.select(
            self.trajs, self.model_devis, self.traj_fmt, self.type_map)
        ms = dpdata.MultiSystems(type_map=self.type_map)
        ms.from_deepmd_npy(confs[0], labeled=False)
        self.assertEqual(len(ms), 1)
        ss = ms[0]
        self.assertEqual(ss.get_nframes(), 1)
        self.assertAlmostEqual(ss['coords'][0][0][1], 2.87, places=2)
        self.assertAlmostEqual(report.candidate_ratio(), 1./3.)
        self.assertAlmostEqual(report.accurate_ratio(), 0./3.)
        self.assertAlmostEqual(report.failed_ratio(), 2./3.)
        
