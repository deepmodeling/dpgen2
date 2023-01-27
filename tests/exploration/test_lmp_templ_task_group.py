import os, textwrap
import numpy as np
import unittest
import itertools

from typing import Set, List
from pathlib import Path

try:
    from exploration.context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.exploration.task import (
    LmpTemplateTaskGroup,
    ExplorationStage,
)
from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    plm_input_name,
)
from unittest.mock import Mock, patch

in_lmp_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd ../graph.003.pb ../graph.001.pb ../graph.002.pb ../graph.000.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)

expected_lmp_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd model.000.pb model.001.pb model.002.pb model.003.pb out_freq 20 out_file model_devi.out
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump all custom 20 traj.dump id type x y z

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)


in_lmp_plm_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd ../graph.003.pb ../graph.001.pb ../graph.002.pb ../graph.000.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

fix             dpgen_plm

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)

expected_lmp_plm_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd model.000.pb model.001.pb model.002.pb model.003.pb out_freq 20 out_file model_devi.out
pair_coeff      * *

fix             dpgen_plm all plumed plumedfile input.plumed outfile output.plumed

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump all custom 20 traj.dump id type x y z

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)

in_plm_template = textwrap.dedent(
    """FOO V_TEMP
DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
RESTRAINT ARG=d1,d2 AT=V_DIST0,bar KAPPA=150.0,150.0 LABEL=restraint
PRINT ARG=restraint.bias
"""
)


class TestLmpTemplateTaskGroup(unittest.TestCase):
    def setUp(self):
        self.lmp_template_fname = Path("lmp.template")
        self.lmp_template_fname.write_text(in_lmp_template)
        self.lmp_plm_template_fname = Path("lmp.plm.template")
        self.lmp_plm_template_fname.write_text(in_lmp_plm_template)
        self.plm_template_fname = Path("plm.template")
        self.plm_template_fname.write_text(in_plm_template)
        self.numb_models = 4
        self.confs = ["foo", "bar"]
        self.lmp_rev_mat = {"V_NSTEPS": [1000], "V_TEMP": [50, 100]}
        self.lmp_plm_rev_mat = {
            "V_NSTEPS": [1000],
            "V_TEMP": [50, 100],
            "V_DIST0": [3, 4],
        }
        self.rev_empty = {}
        self.traj_freq = 20

    def tearDown(self):
        os.remove(self.lmp_template_fname)
        os.remove(self.lmp_plm_template_fname)
        os.remove(self.plm_template_fname)

    def test_lmp(self):
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions=self.lmp_rev_mat,
            traj_freq=self.traj_freq,
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs)
            * len(self.lmp_rev_mat["V_NSTEPS"])
            * len(self.lmp_rev_mat["V_TEMP"]),
        )

        idx = 0
        for cc, ii, jj in itertools.product(
            range(len(self.confs)),
            range(len(self.lmp_rev_mat["V_NSTEPS"])),
            range(len(self.lmp_rev_mat["V_TEMP"])),
        ):
            ee = expected_lmp_template.split("\n")
            ee[0] = ee[0].replace("V_NSTEPS", str(self.lmp_rev_mat["V_NSTEPS"][ii]))
            ee[3] = ee[3].replace("V_TEMP", str(self.lmp_rev_mat["V_TEMP"][jj]))
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                ee,
            )
            idx += 1

    def test_lmp_plm(self):
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_plm_template_fname,
            plm_template_fname=self.plm_template_fname,
            revisions=self.lmp_plm_rev_mat,
            traj_freq=self.traj_freq,
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs)
            * len(self.lmp_plm_rev_mat["V_NSTEPS"])
            * len(self.lmp_plm_rev_mat["V_TEMP"])
            * len(self.lmp_plm_rev_mat["V_DIST0"]),
        )
        idx = 0
        for cc, ii, jj, kk in itertools.product(
            range(len(self.confs)),
            range(len(self.lmp_plm_rev_mat["V_NSTEPS"])),
            range(len(self.lmp_plm_rev_mat["V_TEMP"])),
            range(len(self.lmp_plm_rev_mat["V_DIST0"])),
        ):
            eel = expected_lmp_plm_template.split("\n")
            eel[0] = eel[0].replace(
                "V_NSTEPS", str(self.lmp_plm_rev_mat["V_NSTEPS"][ii])
            )
            eel[3] = eel[3].replace("V_TEMP", str(self.lmp_plm_rev_mat["V_TEMP"][jj]))
            eep = in_plm_template.split("\n")
            eep[0] = eep[0].replace("V_TEMP", str(self.lmp_plm_rev_mat["V_TEMP"][jj]))
            eep[3] = eep[3].replace("V_DIST0", str(self.lmp_plm_rev_mat["V_DIST0"][kk]))
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                eel,
            )
            self.assertEqual(
                task_group[idx].files()[plm_input_name].split("\n"),
                eep,
            )
            idx += 1

    def test_lmp_empty(self):
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions=self.rev_empty,
            traj_freq=self.traj_freq,
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs),
        )
        idx = 0
        for cc in range(len(self.confs)):
            ee = expected_lmp_template.split("\n")
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                ee,
            )
            idx += 1
