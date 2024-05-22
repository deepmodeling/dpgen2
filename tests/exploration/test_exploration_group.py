import os
import textwrap
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
)

import numpy as np

try:
    from exploration.context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from unittest.mock import (
    Mock,
    patch,
)

from dpgen2.constants import (
    calypso_input_file,
    calypso_run_opt_file,
    lmp_conf_name,
    lmp_input_name,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    NPTTaskGroup,
    caly_normalize,
    make_calypso_task_group_from_config,
)

in_template_npt = textwrap.dedent(
    """variable        NSTEPS          equal 1000
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal %f
variable        PRES            equal %f
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box          tilt large
if "${restart} > 0" then "read_restart dpgen.restart.*" else "read_data conf.lmp"
change_box   all triclinic
mass            1 10.000000
mass            2 20.000000
pair_style      deepmd model.000.pb model.001.pb model.002.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}
dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z fx fy fz
restart         10000 dpgen.restart

if "${restart} == 0" then "velocity        all create ${TEMP} 1111"
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.001000
run             ${NSTEPS} upto
"""
)

in_template_nvt = textwrap.dedent(
    """variable        NSTEPS          equal 1000
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal %f
variable        TAU_T           equal 0.100000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box          tilt large
if "${restart} > 0" then "read_restart dpgen.restart.*" else "read_data conf.lmp"
change_box   all triclinic
mass            1 10.000000
mass            2 20.000000
pair_style      deepmd model.000.pb model.001.pb model.002.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}
dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z fx fy fz
restart         10000 dpgen.restart

if "${restart} == 0" then "velocity        all create ${TEMP} 1111"
fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}

timestep        0.001000
run             ${NSTEPS} upto
"""
)


def swap_element(arg):
    bk = arg.copy()
    arg[1] = bk[0]
    arg[0] = bk[1]


ref_input = """NumberOfSpecies = 2
NameOfAtoms = Li La
AtomicNumber = 3 4
NumberOfAtoms = 10 10
PopSize = 30
MaxStep = 5
SystemName = CALYPSO
NumberOfFormula = 1 1
Volume = 0
Ialgo = 2
PsoRatio = 0.6
ICode = 15
NumberOfLbest = 4
NumberOfLocalOptim = 3
Command = sh submit.sh
MaxTime = 9000
GenType = 1
PickUp = False
PickStep = 0
Parallel = F
Split = T
SpeSpaceGroup = 2 230
VSC = F
MaxNumAtom = 100
@DistanceOfIon
1.0 1.0
1.0 1.0
@End
@CtrlRange
1 10
@End
"""


ref_run_opt = """#!/usr/bin/env python3

import os
import time
import glob
import numpy as np

from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter

from deepmd.calculator import DP
'''
structure optimization with DP model and ASE
PSTRESS and fmax should exist in input.dat
'''

def Get_Element_Num(elements):
    '''Using the Atoms.symples to Know Element&Num'''
    element = []
    ele = {}
    element.append(elements[0])
    for x in elements:
        if x not in element :
            element.append(x)
    for x in element:
        ele[x] = elements.count(x)
    return element, ele

def Write_Contcar(contcar, element, ele, lat, pos):
    '''Write CONTCAR'''
    f = open(contcar,'w')
    f.write('ASE-DP-OPT\n')
    f.write('1.0\n')
    for i in range(3):
        f.write('%15.10f %15.10f %15.10f\n' % tuple(lat[i]))
    for x in element:
        f.write(x + '  ')
    f.write('\n')
    for x in element:
        f.write(str(ele[x]) + '  ')
    f.write('\n')
    f.write('Direct\n')
    na = sum(ele.values())
    dpos = np.dot(pos,np.linalg.inv(lat))
    for i in range(na):
        f.write('%15.10f %15.10f %15.10f\n' % tuple(dpos[i]))

def Write_Outcar(outcar, element, ele, volume, lat, pos, ene, force, stress, pstress):
    '''Write OUTCAR'''
    f = open(outcar,'w')
    for x in element:
        f.write('VRHFIN =' + str(x) + '\n')
    f.write('ions per type =')
    for x in element:
        f.write('%5d' % ele[x])
    f.write('\nDirection     XX             YY             ZZ             XY             YZ             ZX\n')
    f.write('in kB')
    f.write('%15.6f' % stress[0])
    f.write('%15.6f' % stress[1])
    f.write('%15.6f' % stress[2])
    f.write('%15.6f' % stress[3])
    f.write('%15.6f' % stress[4])
    f.write('%15.6f' % stress[5])
    f.write('\n')
    ext_pressure = np.sum(stress[0] + stress[1] + stress[2])/3.0 - pstress
    f.write('external pressure = %20.6f kB    Pullay stress = %20.6f  kB\n'% (ext_pressure, pstress))
    f.write('volume of cell : %20.6f\n' % volume)
    f.write('direct lattice vectors\n')
    for i in range(3):
        f.write('%10.6f %10.6f %10.6f\n' % tuple(lat[i]))
    f.write('POSITION                                       TOTAL-FORCE(eV/Angst)\n')
    f.write('-------------------------------------------------------------------\n')
    na = sum(ele.values())
    for i in range(na):
        f.write('%15.6f %15.6f %15.6f' % tuple(pos[i]))
        f.write('%15.6f %15.6f %15.6f\n' % tuple(force[i]))
    f.write('-------------------------------------------------------------------\n')
    f.write('energy  without entropy= %20.6f %20.6f\n' % (ene, ene/na))
    enthalpy = ene + pstress * volume / 1602.17733
    f.write('enthalpy is  TOTEN    = %20.6f %20.6f\n' % (enthalpy, enthalpy/na))

def run_opt(fmax, stress):
    '''Using the ASE&DP to Optimize Configures'''

    calc = DP(model='frozen_model.pb')    # init the model before iteration

    Opt_Step = 1000
    start = time.time()
    # pstress kbar
    pstress = stress
    # kBar to eV/A^3
    # 1 eV/A^3 = 160.21766028 GPa
    # 1 / 160.21766028 ~ 0.006242
    aim_stress = 1.0 * pstress* 0.01 * 0.6242 / 10.0

    poscar_list = sorted(glob.glob("POSCAR_*"), key=lambda x: x.strip("POSCAR_"))
    for poscar in poscar_list:
        to_be_opti = read(poscar)
        to_be_opti.calc = calc
        ucf = UnitCellFilter(to_be_opti, scalar_pressure=aim_stress)
        opt = LBFGS(ucf,trajectory=poscar.strip("POSCAR_") + '.traj')
        opt.run(fmax=fmax,steps=Opt_Step)
        atoms_lat = to_be_opti.cell
        atoms_pos = to_be_opti.positions
        atoms_force = to_be_opti.get_forces()
        atoms_stress = to_be_opti.get_stress()
        # eV/A^3 to GPa
        atoms_stress = atoms_stress/(0.01*0.6242)
        atoms_symbols = to_be_opti.get_chemical_symbols()
        atoms_ene = to_be_opti.get_potential_energy()
        atoms_vol = to_be_opti.get_volume()
        element, ele = Get_Element_Num(atoms_symbols)
        outcar = poscar.replace("POSCAR", "OUTCAR")
        contcar = poscar.replace("POSCAR", "CONTCAR")

        Write_Contcar(contcar, element, ele, atoms_lat, atoms_pos)
        Write_Outcar(outcar, element, ele, atoms_vol, atoms_lat, atoms_pos, atoms_ene, atoms_force, atoms_stress * -10.0, pstress)

    if __name__ == '__main__':
        run_opt(fmax=0.010, stress=0.001)
"""


class TestCPTGroup(unittest.TestCase):
    # def setUp(self):
    #     self.mock_random = Mock()

    @patch("dpgen2.exploration.task.lmp.lmp_input.random")
    def test_npt(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.confs = ["foo", "bar"]
        self.tt = [100, 200]
        self.pp = [1, 10, 100]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model,
            self.mass_map,
            self.tt,
            self.pp,
        )
        cpt_group.set_conf(
            self.confs,
        )
        task_group = cpt_group.make_task()

        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs) * len(self.tt) * len(self.pp))
        for ii in range(ngroup):
            i_idx = ii // (len(self.tt) * len(self.pp))
            j_idx = (ii - len(self.tt) * len(self.pp) * i_idx) // len(self.pp)
            k_idx = ii - len(self.tt) * len(self.pp) * i_idx - len(self.pp) * j_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                self.confs[i_idx],
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_npt % (self.tt[j_idx], self.pp[k_idx]),
            )

    @patch("dpgen2.exploration.task.lmp.lmp_input.random")
    def test_nvt(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.confs = ["foo", "bar"]
        self.tt = [100, 200]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model,
            self.mass_map,
            self.tt,
            ens="nvt",
        )
        cpt_group.set_conf(
            self.confs,
        )
        task_group = cpt_group.make_task()

        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                self.confs[i_idx],
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

    @patch("dpgen2.exploration.task.lmp.lmp_input.random")
    def test_nvt_sample(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.confs = ["foo", "bar"]
        self.tt = [100, 200]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model,
            self.mass_map,
            self.tt,
            ens="nvt",
        )
        cpt_group.set_conf(
            self.confs,
            n_sample=1,
        )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "foo",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "bar",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "foo",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "bar",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

    @patch("dpgen2.exploration.task.npt_task_group.random.shuffle")
    @patch("dpgen2.exploration.task.lmp.lmp_input.random.randrange")
    def test_nvt_sample_random(self, mock_randrange, mock_shuffle):
        mock_randrange.return_value = 1110
        mock_shuffle.side_effect = swap_element
        self.confs = ["foo", "bar"]
        self.tt = [100, 200]
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group = NPTTaskGroup()
        cpt_group.set_md(
            self.numb_model,
            self.mass_map,
            self.tt,
            ens="nvt",
        )
        cpt_group.set_conf(
            self.confs,
            n_sample=1,
            random_sample=True,
        )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "bar",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "foo",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "bar",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )

        task_group = cpt_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs[:1]) * len(self.tt))
        for ii in range(ngroup):
            i_idx = ii // len(self.tt)
            j_idx = ii - len(self.tt) * i_idx
            self.assertEqual(
                task_group[ii].files()[lmp_conf_name],
                "foo",
            )
            self.assertEqual(
                task_group[ii].files()[lmp_input_name],
                in_template_nvt % (self.tt[j_idx]),
            )


class TestCPTStage(unittest.TestCase):
    # def setUp(self):
    #     self.mock_random = Mock()

    @patch("dpgen2.exploration.task.lmp.lmp_input.random")
    def test(self, mock_random):
        mock_random.randrange.return_value = 1110
        self.numb_model = 3
        self.mass_map = [10, 20]

        cpt_group_p = NPTTaskGroup()
        cpt_group_p.set_md(
            self.numb_model,
            self.mass_map,
            [100.0],
            [1.0, 10.0],
        )
        cpt_group_p.set_conf(
            ["foo"],
        )

        cpt_group_t = NPTTaskGroup()
        cpt_group_t.set_md(
            self.numb_model,
            self.mass_map,
            [200.0, 300.0],
            ens="nvt",
        )
        cpt_group_t.set_conf(
            ["bar"],
        )

        stage = ExplorationStage()
        stage.add_task_group(cpt_group_p).add_task_group(cpt_group_t)

        task_group = stage.make_task()

        ngroup = len(task_group)
        self.assertEqual(ngroup, 4)

        ii = 0
        self.assertEqual(task_group[ii].files()[lmp_conf_name], "foo")
        self.assertEqual(
            task_group[ii].files()[lmp_input_name],
            in_template_npt % (100.0, 1.0),
        )
        ii += 1
        self.assertEqual(task_group[ii].files()[lmp_conf_name], "foo")
        self.assertEqual(
            task_group[ii].files()[lmp_input_name],
            in_template_npt % (100.0, 10.0),
        )
        ii += 1
        self.assertEqual(task_group[ii].files()[lmp_conf_name], "bar")
        self.assertEqual(
            task_group[ii].files()[lmp_input_name],
            in_template_nvt % (200.0),
        )
        ii += 1
        self.assertEqual(task_group[ii].files()[lmp_conf_name], "bar")
        self.assertEqual(
            task_group[ii].files()[lmp_input_name],
            in_template_nvt % (300.0),
        )


class TestCalyGroup(unittest.TestCase):
    def setUp(self):
        self.config = caly_normalize(
            {
                "name_of_atoms": ["Li", "La"],
                "numb_of_atoms": [10, 10],
                "numb_of_species": 2,
                "atomic_number": [3, 4],
                "distance_of_ions": [[1.0, 1.0], [1.0, 1.0]],
            }
        )

    def test_caly_task_group_make_task(self):
        tgroup = make_calypso_task_group_from_config(self.config)
        task_group = tgroup.make_task()
        # stage.add_task_group(tasks)

        ngroup = len(task_group)
        self.assertEqual(ngroup, 1)
        self.assertEqual(
            task_group[0].files()[calypso_input_file],
            ref_input,
        )


class TestCalyStage(unittest.TestCase):
    def setUp(self):
        self.config_1 = caly_normalize(
            {
                "name_of_atoms": ["Li", "La"],
                "numb_of_atoms": [10, 10],
                "numb_of_species": 2,
                "atomic_number": [3, 4],
                "distance_of_ions": [[1.0, 1.0], [1.0, 1.0]],
            }
        )
        self.config_2 = caly_normalize(
            {
                "name_of_atoms": ["Li", "H"],
                "numb_of_atoms": [10, 10],
                "numb_of_species": 2,
                "atomic_number": [3, 1],
                "distance_of_ions": [[1.0, 1.0], [1.0, 1.0]],
                "pressure": 100,
                "fmax": 1,
            }
        )
        self.config_random = caly_normalize(
            {
                "name_of_atoms": [
                    ["Li", "La", "Mg", "Al"],
                    ["La", "Be", "Mg", "Ca"],
                    ["H"],
                ],
                "numb_of_atoms": [10, 10, 10],
                "numb_of_species": 3,
                "distance_of_ions": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            }
        )
        self.config_random_dict = caly_normalize(
            {
                "name_of_atoms": [
                    ["Li", "La", "Mg", "Al"],
                    ["La", "Be", "Mg", "Ca"],
                    ["H"],
                ],
                "numb_of_atoms": [10, 10, 10],
                "numb_of_species": 3,
                "distance_of_ions": {"Li": 1.0, "La": 1.1, "H": 0.5},
            }
        )

    def test_00_make_calypso_task(self):
        tgroup_1 = make_calypso_task_group_from_config(self.config_1)
        tgroup_2 = make_calypso_task_group_from_config(self.config_2)

        stage = ExplorationStage()
        stage.add_task_group(tgroup_1).add_task_group(tgroup_2)

        task_group = stage.make_task()

        ngroup = len(task_group)
        self.assertEqual(ngroup, 2)

        self.maxDiff = None
        ii = 0
        self.assertEqual(task_group[ii].files()[calypso_input_file], ref_input)

    def test_01_make_random_calypso_task(self):
        tgroup_1 = make_calypso_task_group_from_config(self.config_random)

        stage = ExplorationStage()
        stage.add_task_group(tgroup_1)

        task_group = stage.make_task()
        self.assertEqual(len(task_group), 1)

    def test_02_make_random_dict_calypso_task(self):
        tgroup_1 = make_calypso_task_group_from_config(self.config_random_dict)

        stage = ExplorationStage()
        stage.add_task_group(tgroup_1)

        task_group = stage.make_task()
        self.assertEqual(len(task_group), 1)
