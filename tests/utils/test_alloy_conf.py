from utils.context import dpgen2
import numpy as np
import unittest, json, shutil, os
import random
import tempfile
import dpdata
from pathlib import Path
from dpgen2.utils.alloy_conf import (
    AlloyConf, 
    generate_alloy_conf_file_content, 
    normalize,
)

class TestAlloyConf(unittest.TestCase):
    def test_concentration_none_vasp(self):
        type_map = ['a', 'b']
        ac = AlloyConf(('sc', 2), type_map, replicate=(2, 2, 1))
        random.seed(0)
        fcs = ac.generate_file_content(3, None, fmt='vasp/poscar')
        sys_list = []
        for ii in fcs:
            tf = Path(tempfile.NamedTemporaryFile().name)
            tf.write_text(ii)
            sys_list.append(dpdata.System(tf, fmt='vasp/poscar'))
            tf.unlink()
        self.assertEqual(len(sys_list), 3)
        for ii in range(3):
            np.testing.assert_array_almost_equal(
                sys_list[ii]['cells'][0],
                np.array([ [4, 0, 0], [0, 4, 0], [0, 0, 2] ])
            )
            np.testing.assert_array_almost_equal(
                np.sort(sys_list[ii]['coords'][0], axis=0),
                np.sort(np.array([ [0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0] ]), axis=0)
            )
            self.assertEqual(sys_list[ii]['atom_names'], ['a', 'b'])
        # result at seed == 0
        np.testing.assert_array_almost_equal(sys_list[0]['atom_types'], [0, 0, 1, 1])
        np.testing.assert_array_almost_equal(sys_list[1]['atom_types'], [0, 0, 1, 1])
        np.testing.assert_array_almost_equal(sys_list[2]['atom_types'], [0, 1, 1, 1])
        self.assertEqual(sys_list[0]['atom_numbs'], [2, 2])
        self.assertEqual(sys_list[1]['atom_numbs'], [2, 2])
        self.assertEqual(sys_list[2]['atom_numbs'], [1, 3])


    def test_concentration_none_lmp(self):
        type_map = ['a', 'b']
        ac = AlloyConf(('sc', 2), type_map, replicate=(2, 2, 1))
        random.seed(0)
        fcs = ac.generate_file_content(3, None, fmt='lammps/lmp')
        sys_list = []
        for ii in fcs:
            tf = Path(tempfile.NamedTemporaryFile().name)
            tf.write_text(ii)
            sys_list.append(dpdata.System(tf, fmt='lammps/lmp'))
            tf.unlink()
        self.assertEqual(len(sys_list), 3)
        for ii in range(3):
            np.testing.assert_array_almost_equal(
                sys_list[ii]['cells'][0],
                np.array([ [4, 0, 0], [0, 4, 0], [0, 0, 2] ])
            )
            np.testing.assert_array_almost_equal(
                np.sort(sys_list[ii]['coords'][0], axis=0),
                np.sort(np.array([ [0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0] ]), axis=0)
            )
        # result at seed == 0
        np.testing.assert_array_almost_equal(sys_list[0]['atom_types'], [1, 1, 0, 0])
        np.testing.assert_array_almost_equal(sys_list[1]['atom_types'], [1, 0, 1, 0])
        np.testing.assert_array_almost_equal(sys_list[2]['atom_types'], [0, 1, 1, 1])
        self.assertEqual(sys_list[0]['atom_numbs'], [2, 2])
        self.assertEqual(sys_list[1]['atom_numbs'], [2, 2])
        self.assertEqual(sys_list[2]['atom_numbs'], [1, 3])


    def test_concentration_list(self):
        type_map = ['a', 'b']
        ac = AlloyConf(('sc', 2), type_map, replicate=(2,2,1))
        random.seed(0)
        fcs = ac.generate_file_content(3, [0., 1.], fmt='lammps/lmp')
        sys_list = []
        for ii in fcs:
            tf = Path(tempfile.NamedTemporaryFile().name)
            tf.write_text(ii)
            sys_list.append(dpdata.System(tf, fmt='lammps/lmp', type_map=type_map))
            tf.unlink()
        self.assertEqual(len(sys_list), 3)
        for ii in range(3):
            np.testing.assert_array_almost_equal(
                sys_list[ii]['cells'][0],
                np.array([ [4, 0, 0], [0, 4, 0], [0, 0, 2] ])
            )
            np.testing.assert_array_almost_equal(
                np.sort(sys_list[ii]['coords'][0], axis=0),
                np.sort(np.array([ [0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0] ]), axis=0)
            )
            self.assertEqual(sys_list[ii]['atom_names'], type_map)
            # result at seed == 0
            np.testing.assert_array_almost_equal(sys_list[ii]['atom_types'], [1, 1, 1, 1])
            self.assertEqual(sys_list[ii]['atom_numbs'], [0, 4])


    def test_concentration_list_list(self):
        type_map = ['a', 'b']
        ac = AlloyConf(('sc', 2), type_map, replicate=(2, 2, 1))
        nframes = 5
        random.seed(0)
        fcs = ac.generate_file_content(nframes, [[0., 1.], [1., 0.]], fmt='lammps/lmp')
        sys_list = []
        for ii in fcs:
            tf = Path(tempfile.NamedTemporaryFile().name)
            tf.write_text(ii)
            sys_list.append(dpdata.System(tf, fmt='lammps/lmp'))
            tf.unlink()
        self.assertEqual(len(sys_list), nframes)
        for ii in range(nframes):
            np.testing.assert_array_almost_equal(
                sys_list[ii]['cells'][0],
                np.array([ [4, 0, 0], [0, 4, 0], [0, 0, 2] ])
            )
            np.testing.assert_array_almost_equal(
                np.sort(sys_list[ii]['coords'][0], axis=0),
                np.sort(np.array([ [0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0] ]), axis=0)
            )
        # result at seed == 0
        np.testing.assert_array_almost_equal(sys_list[0]['atom_types'], [0, 0, 0, 0])
        np.testing.assert_array_almost_equal(sys_list[1]['atom_types'], [0, 0, 0, 0])
        np.testing.assert_array_almost_equal(sys_list[2]['atom_types'], [1, 1, 1, 1])
        np.testing.assert_array_almost_equal(sys_list[3]['atom_types'], [0, 0, 0, 0])
        np.testing.assert_array_almost_equal(sys_list[4]['atom_types'], [1, 1, 1, 1])
        self.assertEqual(sys_list[0]['atom_numbs'], [4, 0])
        self.assertEqual(sys_list[1]['atom_numbs'], [4, 0])
        self.assertEqual(sys_list[2]['atom_numbs'], [0, 4])
        self.assertEqual(sys_list[3]['atom_numbs'], [4, 0])
        self.assertEqual(sys_list[4]['atom_numbs'], [0, 4])


    def test_concentration_list_sys(self):
        type_map = ['a', 'b']
        ac = AlloyConf(('sc', 2), type_map, replicate=(2, 2, 1))
        random.seed(0)
        sys_list = ac.generate_systems(3, [0., 1.])
        self.assertEqual(len(sys_list), 3)
        for ii in range(3):
            np.testing.assert_array_almost_equal(
                sys_list[ii]['cells'][0],
                np.array([ [4, 0, 0], [0, 4, 0], [0, 0, 2] ])
            )
            np.testing.assert_array_almost_equal(
                np.sort(sys_list[ii]['coords'][0], axis=0),
                np.sort(np.array([ [0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0] ]), axis=0)
            )
            self.assertEqual(sys_list[ii]['atom_names'], type_map)
            # result at seed == 0
            np.testing.assert_array_almost_equal(sys_list[ii]['atom_types'], [1, 1, 1, 1])
            self.assertEqual(sys_list[ii]['atom_numbs'], [0, 4])


    def test_concentration_none_vasp_in_one(self):
        type_map = ['a', 'b']
        random.seed(0)
        fcs = generate_alloy_conf_file_content(
            ('sc', 2),
            type_map,
            3,
            replicate=(2,2,1),
            concentration=None,
            fmt='vasp/poscar',
        )
        sys_list = []
        for ii in fcs:
            tf = Path(tempfile.NamedTemporaryFile().name)
            tf.write_text(ii)
            sys_list.append(dpdata.System(tf, fmt='vasp/poscar'))
            tf.unlink()
        self.assertEqual(len(sys_list), 3)
        for ii in range(3):
            np.testing.assert_array_almost_equal(
                sys_list[ii]['cells'][0],
                np.array([ [4, 0, 0], [0, 4, 0], [0, 0, 2] ])
            )
            np.testing.assert_array_almost_equal(
                np.sort(sys_list[ii]['coords'][0], axis=0),
                np.sort(np.array([ [0, 0, 0], [0, 2, 0], [2, 0, 0], [2, 2, 0] ]), axis=0)
            )
            self.assertEqual(sys_list[ii]['atom_names'], ['a', 'b'])
        # result at seed == 0
        np.testing.assert_array_almost_equal(sys_list[0]['atom_types'], [0, 0, 1, 1])
        np.testing.assert_array_almost_equal(sys_list[1]['atom_types'], [0, 0, 1, 1])
        np.testing.assert_array_almost_equal(sys_list[2]['atom_types'], [0, 1, 1, 1])
        self.assertEqual(sys_list[0]['atom_numbs'], [2, 2])
        self.assertEqual(sys_list[1]['atom_numbs'], [2, 2])
        self.assertEqual(sys_list[2]['atom_numbs'], [1, 3])


    def test_normalize(self):
        in_data = {
            "lattice" : ("bcc", 2),
            "type_map" : ['a', 'b'],
        }
        expected_out_data = {
            "lattice" : ("bcc", 2),
            "type_map" : ['a', 'b'],
            "replicate" : None,
            "numb_confs" : 1,
            "cell_pert_frac" : 0.,
            "atom_pert_dist" : 0.,
            'concentration' : None,
            "fmt" : "lammps/lmp",
        }
        out_data = normalize(
            in_data,
        )
        self.assertEqual(out_data, expected_out_data)
