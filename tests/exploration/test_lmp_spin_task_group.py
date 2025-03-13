import itertools
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
    lmp_conf_name,
    lmp_input_name,
    plm_input_name,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    LmpSpinTaskGroup,
)

in_lmp_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000
variable        MASS            equal V_MASS

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepspin model.000.pth model.001.pth model.002.pth model.003.pth out_freq ${THERMO_FREQ}
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P} mass ${MASS}

timestep        0.002000
run             

compute         mag   all  spin
compute         pe    all  pe
compute         ke    all  ke
compute         temp  all  temp
compute         spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz

thermo          10
thermo_style    custom step time v_magnorm temp v_tmag press etotal ke pe v_emag econserve

dump            dpgen_dump all custom 10 traj.dump id type x y z c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7] c_spin[8] c_spin[9] c_spin[10]
dump_modify     dpgen_dump sort id

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
variable        MASS            equal V_MASS

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepspin model.000.pth model.001.pth model.002.pth model.003.pth out_freq ${THERMO_FREQ}
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P} mass ${MASS}

timestep        0.002000
run             

compute         mag   all  spin
compute         pe    all  pe
compute         ke    all  ke
compute         temp  all  temp
compute         spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz

thermo          10
thermo_style    custom step time v_magnorm temp v_tmag press etotal ke pe v_emag econserve

dump            dpgen_dump all custom 10 traj.dump id type x y z c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7] c_spin[8] c_spin[9] c_spin[10]
dump_modify     dpgen_dump sort id

run             ${NSTEPS}
"""
)


class TestLmpSpinTaskGroup(unittest.TestCase):
    def setUp(self):
        self.lmp_template_fname = Path("lmp.template")
        self.lmp_template_fname.write_text(in_lmp_template)
        self.numb_models = 8
        self.confs = ["foo", "bar"]
        self.lmp_rev_mat = {
            "V_NSTEPS": [1000],
            "V_TEMP": [50, 100],
            "V_MASS": [0.01, 0.1],
        }
        self.rev_empty = {}

    def tearDown(self):
        os.remove(self.lmp_template_fname)

    def test_lmp(self):
        task_group = LmpSpinTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models, self.lmp_template_fname, revisions=self.lmp_rev_mat
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs)
            * len(self.lmp_rev_mat["V_NSTEPS"])
            * len(self.lmp_rev_mat["V_TEMP"])
            * len(self.lmp_rev_mat["V_MASS"]),
        )

        idx = 0
        for cc, ii, jj, kk in itertools.product(
            range(len(self.confs)),
            range(len(self.lmp_rev_mat["V_NSTEPS"])),
            range(len(self.lmp_rev_mat["V_TEMP"])),
            range(len(self.lmp_rev_mat["V_MASS"])),
        ):
            ee = expected_lmp_template.split("\n")
            ee[0] = ee[0].replace("V_NSTEPS", str(self.lmp_rev_mat["V_NSTEPS"][ii]))
            ee[3] = ee[3].replace("V_TEMP", str(self.lmp_rev_mat["V_TEMP"][jj]))
            ee[7] = ee[7].replace("V_MASS", str(self.lmp_rev_mat["V_MASS"][kk]))
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                ee,
            )
            idx += 1

    def test_lmp_empty(self):
        task_group = LmpSpinTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models, self.lmp_template_fname, revisions=self.rev_empty
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
